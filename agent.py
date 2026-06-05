import os
import datetime
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Kubernetes & LangChain
from kubernetes import client, config
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, AIMessage
from langchain_core.tools import tool

# LangGraph Core
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent
# ---------------------------------------------------------
# 1. SETUP & KUBERNETES AUTH
# ---------------------------------------------------------
load_dotenv()
try:
    config.load_kube_config()
    print("✅ Connected to Kubernetes Cluster!")
except Exception as e:
    print(f"❌ Error connecting to K8s: {e}")
    exit(1)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ---------------------------------------------------------
# 2. READ-ONLY TOOLBOX (For the Investigator)
# ---------------------------------------------------------
@tool
def get_k8s_pod_status(namespace: str = "default") -> str:
    """Fetches the current status of all Kubernetes pods."""
    v1 = client.CoreV1Api()
    try:
        pods = v1.list_namespaced_pod(namespace)
        return "\n".join([f"Pod: {p.metadata.name} | Status: {p.status.phase}" for p in pods.items])
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_k8s_pod_events(pod_name: str, namespace: str = "default") -> str:
    """Fetches recent K8s events for a pod to diagnose ImagePullBackOff or Scheduling errors."""
    v1 = client.CoreV1Api()
    try:
        events = v1.list_namespaced_event(namespace, field_selector=f"involvedObject.name={pod_name}")
        if not events.items: return "No events found."
        return "\n".join([f"Reason: {e.reason} | Message: {e.message}" for e in events.items[-5:]])
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_k8s_logs(pod_name: str, namespace: str = "default") -> str:
    """Fetches the last 50 lines of logs for a pod. Use this if a pod is CrashLooping or has Application Errors."""
    v1 = client.CoreV1Api()
    try:
        logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=50)
        return logs[-2000:] # Limit string size to prevent token blowouts
    except Exception as e:
        return f"Error fetching logs: {str(e)}"

# ---------------------------------------------------------
# 3. PYDANTIC SCHEMA (Forces LLM to output perfect JSON)
# ---------------------------------------------------------
class RemediationPlan(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning of why this fix is chosen based on the logs/events.")
    action: str = Field(description="Must be exactly one of: 'scale', 'restart', 'update_image', or 'none'")
    deployment_name: str = Field(description="The exact name of the K8s deployment to modify (e.g. 'cartservice')")
    container_name: Optional[str] = Field(default="", description="If updating image, the container name")
    new_image: Optional[str] = Field(default="", description="If updating image, the new docker image tag")
    replicas: Optional[int] = Field(default=1, description="If scaling, the new number of replicas")

# ---------------------------------------------------------
# 4. GRAPH STATE DEFINITION
# ---------------------------------------------------------
class SREState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    diagnosis: str
    action_plan: dict
    human_approved: bool

# ---------------------------------------------------------
# 5. LANGGRAPH NODES
# ---------------------------------------------------------
def investigator_node(state: SREState):
    print("\n[🔍 NODE: INVESTIGATOR] Scanning cluster metrics, logs, and events...")
    query = state["messages"][-1].content
    
    # The investigator is an internal sub-agent that loops through read-only tools
    investigator_agent = create_react_agent(llm, [get_k8s_pod_status, get_k8s_pod_events, get_k8s_logs])
    result = investigator_agent.invoke({"messages": [("user", query)]})
    
    diagnosis = result["messages"][-1].content
    return {"diagnosis": diagnosis}

def planner_node(state: SREState):
    print("[🧠 NODE: PLANNER] Analyzing diagnosis and drafting JSON remediation plan...")
    diagnosis = state["diagnosis"]
    
    prompt = f"""
    You are an Expert Kubernetes SRE. Read this diagnosis:
    {diagnosis}
    
    Decide the exact Kubernetes action needed to fix the issue. 
    SOP RULES:
    1. If a pod has 'ImagePullBackOff' or 'ErrImagePull', you MUST use the 'update_image' action. Restarting will not fix a broken image!
    2. If a pod is overwhelmed with traffic, use the 'scale' action.
    3. If a pod is deadlocked or CrashLooping (but the image is correct), use 'restart'.
    4. If the cluster is completely healthy, set action to 'none'.
    """
    
    structured_llm = llm.with_structured_output(RemediationPlan)
    plan = structured_llm.invoke(prompt)
    
    # Using model_dump() fixes the Pydantic V2 warning!
    return {"action_plan": plan.model_dump()}

def approval_node(state: SREState):
    plan = state["action_plan"]
    
    if plan["action"] == "none":
        print("\n[✅ CLUSTER HEALTHY] No remediation needed.")
        return {"human_approved": False}
        
    print("\n" + "="*60)
    print("🚨 [NODE: HUMAN GATEKEEPER - APPROVAL REQUIRED] 🚨")
    print(f"➔ Reasoning: {plan['reasoning']}")
    print(f"➔ Proposed Action: {plan['action'].upper()} on deployment '{plan['deployment_name']}'")
    
    if plan["action"] == "scale":
        print(f"   - Target Replicas: {plan['replicas']}")
    elif plan["action"] == "update_image":
        print(f"   - Target Image: {plan['new_image']}")
    print("="*60)
    
    ans = input("Do you approve this production change? (y/n): ")
    return {"human_approved": ans.lower() == 'y'}

def executor_node(state: SREState):
    if not state["human_approved"]:
        print("[🛑 NODE: EXECUTOR] Operation aborted by human.")
        return {}

    print("[⚙️ NODE: EXECUTOR] Applying patch directly to Kubernetes API...")
    plan = state["action_plan"]
    action = plan["action"]
    name = plan["deployment_name"]
    namespace = "default"
    apps_v1 = client.AppsV1Api()
    
    # Pure Python Execution (Zero LLM Hallucination Risk here)
    try:
        if action == "restart":
            patch = {"spec": {"template": {"metadata": {"annotations": {"restartedAt": datetime.datetime.now().isoformat()}}}}}
            apps_v1.patch_namespaced_deployment(name, namespace, patch)
            print(f"✅ SUCCESS: Restarted deployment '{name}'")
            
        elif action == "scale":
            patch = {"spec": {"replicas": plan["replicas"]}}
            apps_v1.patch_namespaced_deployment_scale(name, namespace, patch)
            print(f"✅ SUCCESS: Scaled '{name}' to {plan['replicas']} replicas")
            
        elif action == "update_image":
            deployment = apps_v1.read_namespaced_deployment(name, namespace)
            for c in deployment.spec.template.spec.containers:
                if c.name == plan["container_name"]:
                    c.image = plan["new_image"]
            apps_v1.patch_namespaced_deployment(name, namespace, deployment)
            print(f"✅ SUCCESS: Updated image for '{name}'")
            
    except Exception as e:
        print(f"❌ K8s Execution Error: {e}")
        
    return {}

# ---------------------------------------------------------
# 6. BUILD THE STATEGRAPH
# ---------------------------------------------------------
builder = StateGraph(SREState)

# Add all our custom nodes
builder.add_node("investigate", investigator_node)
builder.add_node("plan", planner_node)
builder.add_node("approve", approval_node)
builder.add_node("execute", executor_node)

# Define the exact execution flow
builder.set_entry_point("investigate")
builder.add_edge("investigate", "plan")
builder.add_edge("plan", "approve")
builder.add_edge("approve", "execute")
builder.add_edge("execute", END)

graph = builder.compile()

# ---------------------------------------------------------
# 7. RUN THE SYSTEM
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Waking up AI SRE Orchestrator...")
    
    query = """
    Check the cluster status.
    If cartservice has a broken image, fix it (cartservice:v0.8.0, container 'server').
    If frontend is healthy but struggling, consider restarting it or scaling it. 
    Find the root cause of any issue and plan a fix.
    """
    
    # Execute the Graph!
    graph.invoke({"messages": [("user", query)]})
    print("\n🏁 Graph Execution Complete.")