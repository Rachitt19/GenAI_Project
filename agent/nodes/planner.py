import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from agent.utils.llm import get_llm

def generate_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    reasoning = state.get('reasoning', {})
    
    system_msg = SystemMessage(content='''You are an elite EV Infrastructure Planner.
Translate the structured reasoning into a high-quality actionable plan.
You MUST output ONLY a valid JSON object with this EXACT schema:
{
  "high_load_locations": ["string"],
  "recommendations": [
    {
      "type": "charger_expansion | load_balancing | scheduling",
      "location": "string",
      "action": "clear specific step you are recommending",
      "justification": "explicit logical foundation",
      "priority": "high | medium | low"
    }
  ],
  "risk_level": "Low | Medium | High",
  "confidence_score": 0.95,
  "alternative_strategies": ["string"]
}
RULES:
- Recommendations must be specific and actionable.
- No generic text allowed.
- Must use reasoning outputs explicitly.
- DO NOT output any markdown tags or text outside of the JSON.''')

    user_content = f"Reasoning Data:\n{json.dumps(reasoning)}\n\nGenerate the structured JSON plan."
    user_msg = HumanMessage(content=user_content)
    
    try:
        llm = get_llm()
        result = llm.invoke([system_msg, user_msg])
        
        content = result.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        plan = json.loads(content.strip())
        
        if "confidence_score" not in plan:
            plan["confidence_score"] = float(reasoning.get("confidence", 0.5))
    except Exception as e:
        plan = {
            "high_load_locations": ["Unknown Core System"],
            "recommendations": [
                {
                    "type": "load_balancing",
                    "location": "System Wide",
                    "action": "Halt deployment and deploy manual review",
                    "justification": f"Parsing failed natively: {str(e)}",
                    "priority": "high"
                }
            ],
            "risk_level": state.get("insights", {}).get("risk_level", "Unknown"),
            "confidence_score": 0.1,
            "alternative_strategies": ["Manual Audit and Human Review"]
        }
    
    return {"final_plan": plan}
