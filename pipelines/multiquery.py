from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from config import llm
from routing import get_retriever_for_query
from langchain.load import dumps, loads

# Multiquery prompt to generate alternative queries
template_multi = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.
Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template_multi)

# Chain for generating alternative queries
def generate_alternate_queries(question: str):
    chain = (
        prompt_perspectives 
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    return chain.invoke({"question": question})

def get_unique_union(documents: list[list]):
    """Returns the unique union of retrieved documents."""
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

def format_docs(docs):
    """Formats list of Document objects into a string."""
    return "\n\n".join(doc.page_content for doc in docs)

# RAG prompt for synthesizing the final answer
rag_template = """Answer the following question based on this context (avoid saying "based on the context provided"):

{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

def run_multiquery_pipeline(question: str):
    # Generate alternative queries
    alternate_queries = generate_alternate_queries(question)

    all_retrieved_docs = []
    for alt_query in alternate_queries:
        current_retriever = get_retriever_for_query(alt_query)
        docs = current_retriever.invoke(alt_query)
        all_retrieved_docs.append(docs)

    final_docs = get_unique_union(all_retrieved_docs)
    # Prepare final RAG chain
    final_rag_chain = (
        {
            "context": RunnableLambda(lambda x: format_docs(x["docs"])),
            "question": lambda x: x["question"]
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    final_answer = final_rag_chain.invoke({
        "docs": final_docs,
        "question": question
    })
    return final_answer
