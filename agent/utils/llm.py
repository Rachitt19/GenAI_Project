from langchain_openai import ChatOpenAI
from agent.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_NAME

def get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.2,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL
    )
