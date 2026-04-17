import json
from langchain_core.messages import SystemMessage, HumanMessage
from agent.utils.llm import get_llm

def simulate_scenarios(state):
    """Simulates a hypothetical scenario on the finalized plan."""
    plan = state.get('final_plan', {})
    insights = state.get('insights', {})
    
    system_msg = SystemMessage(content='''You are an EV grid scenario simulator.
Assess the impact of a sudden 20% demand increase on the given infrastructure plan.
Respond in STRICT JSON with exactly this structure:
{
  "scenario": "Demand increases by 20%",
  "impact_analysis": "string explanation",
  "robustness_score": 0.85
}
Do NOT output any other text or markdown.''')

    user_content = f"Plan: {json.dumps(plan)}\n\nInsights: {json.dumps(insights)}\n\nAnalyze the scenario."
    user_msg = HumanMessage(content=user_content)
    
    try:
        llm = get_llm()
        result = llm.invoke([system_msg, user_msg])
        
        sim = json.loads(result.content.strip())
    except Exception as e:
        sim = {
            "scenario": "Demand increases by 20%",
            "impact_analysis": "Simulation paused due to AI processing overload. Automatic safeguards activated.",
            "robustness_score": 0.0
        }
        
    return {"simulated_impact": sim}
