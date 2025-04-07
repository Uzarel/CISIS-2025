from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from config import llm
from routing import get_retriever_for_query

# Decomposition prompt to generate sub-questions
template_decomposition = """You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
Generate multiple search queries related to: {question}
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template_decomposition)

def generate_sub_questions(question: str):
    chain = (
        prompt_decomposition 
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    return chain.invoke({"question": question})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_qa_pair(question, answer):
    return f"Question: {question}\nAnswer: {answer}\n"

# RAG prompt
rag_template = """Answer the following question based on this context (avoid saying "based on the context provided"):

{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

def run_decomposition_pipeline(question: str):
    sub_questions = generate_sub_questions(question)
    qa_pairs = ""
    for sub_q in sub_questions:
        current_retriever = get_retriever_for_query(sub_q)
        sub_docs = current_retriever.invoke(sub_q)

        rag_chain_decomp = (
            RunnableLambda(lambda x: {
                "context": format_docs(x["docs"]),
                "question": x["question"],
                "q_a_pairs": x["q_a_pairs"]
            })
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain_decomp.invoke({
            "question": sub_q,
            "q_a_pairs": qa_pairs,
            "docs": sub_docs
        })

        qa_pairs += "\n---\n" + format_qa_pair(sub_q, answer)

    final_context_decomp = qa_pairs.strip()

    # Synthesis phase
    synthesis_template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""
    synthesis_prompt = ChatPromptTemplate.from_template(synthesis_template)

    final_rag_chain_decomp = synthesis_prompt | llm | StrOutputParser()
    final_answer = final_rag_chain_decomp.invoke({
        "context": final_context_decomp,
        "question": question
    })
    return final_answer
