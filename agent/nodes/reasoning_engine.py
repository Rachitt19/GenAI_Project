import json
from typing import Dict, Any, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from agent.utils.llm import get_llm

def validate_reasoning(data: Dict[str, Any]) -> Tuple[bool, str]:
    required_keys = {"observations", "inferences", "decisions", "uncertainties", "confidence"}
    
    if not isinstance(data, dict):
        return False, "Output must be a JSON dictionary."
        
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        return False, f"Missing required keys: {', '.join(missing_keys)}"
        
    for key in ["observations", "inferences", "decisions", "uncertainties"]:
        if not isinstance(data[key], list):
            return False, f"'{key}' must be a list of strings."
            
    if not isinstance(data["confidence"], (int, float)):
        return False, "'confidence' must be a numeric value between 0 and 1."
        
    if len(data["observations"]) < 3:
        return False, "Must provide at least 3 specific measurable observations."
        
    if len(data["inferences"]) < 2:
        return False, "Must provide at least 2 logical inferences."
        
    if len(data["decisions"]) < 2:
        return False, "Must provide at least 2 clear, actionable decisions."
        
    for dec in data["decisions"]:
        if len(str(dec)) < 15 or "optimize" in str(dec).lower() and len(str(dec)) < 25:
            return False, f"Decision '{dec}' is too vague. Must include specific location, time, or measurable action."
            
    return True, "Valid"

def do_reasoning(state: Dict[str, Any]) -> Dict[str, Any]:
    insights = state.get('insights', {})
    patterns = state.get('patterns', {})
    knowledge = state.get('retrieved_knowledge', [])
    feedback = state.get('feedback', "")
    
    system_msg = SystemMessage(content='''You are an elite AI systems architect deciding on EV charging infrastructure.
Analyze the patterns and metrics to generate deep, structured reasoning.
You MUST output ONLY valid JSON using this EXACT schema:
{
  "observations": ["measurable metric 1", "metric 2", "metric 3"],
  "inferences": ["logical conclusion 1", "logical conclusion 2"],
  "decisions": ["actionable decision 1 with location/time", "actionable decision 2"],
  "uncertainties": ["known limitations"],
  "confidence": 0.85
}
RULES:
- Provide at least 3 specific measurable observations.
- Provide at least 2 logical inferences.
- Provide at least 2 decisions.
- DO NOT use vague statements like 'optimize system'. Be hyper-specific indicating locations, times, or thresholds.
- DO NOT output any markdown tags or text outside of the JSON.''')

    user_content = f"""Data Insights:
{json.dumps(insights)}

Patterns:
{json.dumps(patterns)}

Knowledge Guidelines:
{json.dumps(knowledge)}

Feedback from Previous Iterations (Address this logically):
{feedback}

Generate structured reasoning strictly following the JSON format."""

    user_msg = HumanMessage(content=user_content)
    
    max_retries = 2
    llm = get_llm()
    
    for attempt in range(max_retries + 1):
        try:
            result = llm.invoke([system_msg, user_msg])
            content = result.content.strip()
            
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            reasoning_dict = json.loads(content.strip())
            
            is_valid, validation_msg = validate_reasoning(reasoning_dict)
            if is_valid:
                return {"reasoning": reasoning_dict}
            else:
                user_msg = HumanMessage(content=user_content + f"\n\nERROR IN PREVIOUS ATTEMPT: {validation_msg}\nPlease fix and strictly follow the schema and quality rules.")
                
        except Exception as e:
            user_msg = HumanMessage(content=user_content + f"\n\nERROR PARSING JSON: {str(e)}\nEnsure you ONLY output valid JSON.")
    
    # Exhausted retries, fallback
    fallback = {
        "observations": ["High demand detected", "Retry iteration limit reached natively", "Usage requires infrastructure oversight"],
        "inferences": ["System under load stress", "Manual audit required due to reasoning anomaly"],
        "decisions": ["Implement dynamic pricing buffers", "Alert local station operator immediately"],
        "uncertainties": ["LLM formatting or reasoning repeatedly failed", "Granular peak times obscure"],
        "confidence": 0.3
    }
    return {"reasoning": fallback}
