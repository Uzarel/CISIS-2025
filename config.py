from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# Default settings for LLM and embeddings
LLM_MODEL = "gemma3:27b"
HAS_REASONING = False # TODO: Implement custom logic for invoking LLM with reasoning capabilities, i.e. exclude <think></think> content
EMBEDDINGS_MODEL = "nomic-embed-text"
DEFAULT_TEMPERATURE = 0

# Define the LLM and embeddings
llm = OllamaLLM(model=LLM_MODEL, temperature=DEFAULT_TEMPERATURE)
embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
