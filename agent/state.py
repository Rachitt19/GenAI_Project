from typing import TypedDict, Dict, Any, List
import pandas as pd

class AgentState(TypedDict):
    df_raw: Any # pd.DataFrame
    insights: Dict[str, Any]
    patterns: Dict[str, Any]
    retrieved_knowledge: List[str]
    reasoning: Dict[str, Any]
    final_plan: Dict[str, Any]
    feedback: str # used for iterative planning loop
    iteration_count: int # tracks the evaluation loops
    simulated_impact: Dict[str, Any] # stores output of what-if scenario
