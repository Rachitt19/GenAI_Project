from typing import Dict, Any

def evaluate_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates the generated plan and provides strict, constructive feedback."""
    plan = state.get('final_plan', {})
    confidence = plan.get('confidence_score', 0.0)
    recommendations = plan.get('recommendations', [])
    reasoning = state.get('reasoning', {})
    iters = state.get('iteration_count', 0)
    
    issues = []
    
    if confidence < 0.75:
        issues.append(f"Confidence score is weak ({confidence}). You must investigate uncertainties deeper and cross-index retrieved knowledge.")
        
    if not recommendations:
        issues.append("Plan completely lacks actionable recommendations. Must provide at least one specific deployment step.")
        
    for rec in recommendations:
        action = rec.get("action", "")
        if len(action) < 15 or "optimize" in action.lower():
            issues.append(f"Recommendation action '{action}' is too vague. You must specify exact parameters (like load thresholds, times, or exact hardware placement).")
            
    observations = reasoning.get("observations", [])
    if not observations or len(observations) < 3:
        issues.append("Missing or inadequate analytical density. Reasoning MUST contain at least 3 concrete data observations from patterns to found decisions safely.")
        
    if issues:
        feedback = "Plan evaluation identified the following unacceptable weaknesses leading to rejection: " + " | ".join(issues)
    else:
        feedback = "Plan passes final analysis. Specific, actionable, and mathematically logical."
        
    return {"feedback": feedback, "iteration_count": iters + 1}
