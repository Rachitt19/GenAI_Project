from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes.demand_analyzer import analyze_demand
from agent.nodes.deep_analysis import deep_analyze_demand
from agent.nodes.pattern_detector import detect_patterns
from agent.nodes.rag_retriever import retrieve_knowledge
from agent.nodes.reasoning_engine import do_reasoning
from agent.nodes.planner import generate_plan
from agent.nodes.evaluator import evaluate_plan
from agent.nodes.simulator import simulate_scenarios

def route_demand(state: AgentState):
    if state.get("insights", {}).get("risk_level") == "High":
        return "deep_analysis"
    return "pattern_detection"

def route_evaluation(state: AgentState):
    iters = state.get("iteration_count", 0)
    plan = state.get("final_plan", {})
    confidence = plan.get("confidence_score", 1.0)
    feedback = state.get("feedback", "")
    
    # 8. EARLY EXIT
    if confidence > 0.90 and "weak" not in feedback.lower():
        return END
        
    # 7. IMPROVE LOOP CONTROL (confidence < 0.75 or feedback contains 'weak')
    if (confidence < 0.75 or "weak" in feedback.lower()) and iters < 3:
        return "reasoning"
        
    # Standard Flow Simulation
    return "simulate_scenarios"

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("demand_analysis", analyze_demand)
    workflow.add_node("deep_analysis", deep_analyze_demand)
    workflow.add_node("pattern_detection", detect_patterns)
    workflow.add_node("rag_retrieval", retrieve_knowledge)
    workflow.add_node("reasoning", do_reasoning)
    workflow.add_node("planning", generate_plan)
    workflow.add_node("evaluate_plan", evaluate_plan)
    workflow.add_node("simulate_scenarios", simulate_scenarios)
    
    workflow.add_edge(START, "demand_analysis")
    
    workflow.add_conditional_edges(
        "demand_analysis",
        route_demand,
        {"deep_analysis": "deep_analysis", "pattern_detection": "pattern_detection"}
    )
    workflow.add_edge("deep_analysis", "pattern_detection")
    
    workflow.add_edge("pattern_detection", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "reasoning")
    workflow.add_edge("reasoning", "planning")
    workflow.add_edge("planning", "evaluate_plan")
    
    workflow.add_conditional_edges(
        "evaluate_plan",
        route_evaluation,
        {
            "reasoning": "reasoning",
            "simulate_scenarios": "simulate_scenarios",
            END: END
        }
    )
    
    workflow.add_edge("simulate_scenarios", END)
    
    app = workflow.compile()
    return app
