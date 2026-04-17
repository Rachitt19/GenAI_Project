from langchain_openai import ChatOpenAI
from agent.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_NAME

def get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.2,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "GenAI Dashboard"}
    )
