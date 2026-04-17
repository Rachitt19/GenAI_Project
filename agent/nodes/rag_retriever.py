from agent.utils.vector_store import get_retriever

def retrieve_knowledge(state):
    insights = state.get('insights', {})
    patterns = state.get('patterns', {})
    
    query = (
        f"Risk Level: {insights.get('risk_level', 'Unknown')}. "
        f"Peak Hours: {insights.get('peak_hours', [])}. "
        f"Repeated Congestion: {patterns.get('repeated_congestion', 'Unknown')}. "
        f"Grid Impact: {patterns.get('grid_stability_impact', 'Unknown')}. "
    )
    
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        retrieved_content = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        retrieved_content = f"Could not retrieve knowledge due to error: {str(e)}"
        
    return {"retrieved_knowledge": retrieved_content}
