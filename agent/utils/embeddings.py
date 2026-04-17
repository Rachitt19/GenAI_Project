from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model(model_name="all-MiniLM-L6-v2"):
    """
    Returns an efficient, open-source embedding model that runs locally
    without needing any API keys.
    """
    return HuggingFaceEmbeddings(model_name=model_name)
