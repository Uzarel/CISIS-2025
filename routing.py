from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm
from indexing import retriever_dict

# Define the routing prompt template
routing_template = """You are a classifier that determines whether a legal question is about Italian inheritance laws or Italian divorce laws.
Based on the question, answer with JUST one of the following categories:
- INHERITANCE (for questions related to inheritance law)
- DIVORCE (for questions related to divorce law)
Question: {question}
Category:"""

routing_prompt = ChatPromptTemplate.from_template(routing_template)

def route_query(query: str) -> str:
    """Determine the category of a query using the router."""
    routing_chain = routing_prompt | llm | StrOutputParser()
    category = routing_chain.invoke({"question": query}).strip().upper()
    return category.split()[0]

def get_retriever_for_query(query: str):
    category = route_query(query)
    # Fallback to inheritance_retriever if no valid category is returned
    return retriever_dict.get(category, retriever_dict["INHERITANCE"])
