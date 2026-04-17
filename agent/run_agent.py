from agent.graph import build_graph

def run_planning_agent(df):
    app = build_graph()
    initial_state = {
        "df_raw": df,
        "insights": {},
        "patterns": {},
        "retrieved_knowledge": [],
        "reasoning": {},
        "final_plan": {},
        "feedback": "",
        "iteration_count": 0,
        "simulated_impact": {}
    }
    
    # Run the compiled LangGraph execution
    result = app.invoke(initial_state)
    return result
