from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from config import llm
from routing import get_retriever_for_query

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Stepback paraphrasing prompt
step_back_template = """You are an expert at Italian law. Your task is to step back and paraphrase a legal question to a more generic form that is easier to answer.
Examples:
Input: What are the rules for inheritance in Italy?
Output: what are the general principles of Italian inheritance law?

Input: How does divorce work in Italy?
Output: what are the main aspects of Italian divorce law?

Now, step back the following question:
{question}"""
prompt_step_back = ChatPromptTemplate.from_template(step_back_template)

def generate_step_back_query(question: str):
    chain = (
        prompt_step_back
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"question": question})

# Synthesis prompt for final answer
response_prompt_template = """You are an expert in Italian law. Your response should be comprehensive and consistent with the following contexts if they are relevant:

# Normal Context:
{normal_context}

# Step-Back Context:
{step_back_context}

Original Question: {question}
Answer (avoid saying "based on the context provided"):"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

def run_stepback_pipeline(question: str):
    # Generate stepback version of the question
    step_back_query = generate_step_back_query(question)

    normal_retriever = get_retriever_for_query(question)
    step_back_retriever = get_retriever_for_query(step_back_query)

    normal_context = normal_retriever.invoke(question)
    step_back_context = step_back_retriever.invoke(step_back_query)

    chain_step_back = (
        RunnableLambda(lambda x: {
            "normal_context": format_docs(x["normal_context"]),
            "step_back_context": format_docs(x["step_back_context"]),
            "question": x["question"]
        })
        | response_prompt
        | llm
        | StrOutputParser()
    )

    final_answer = chain_step_back.invoke({
        "normal_context": normal_context,
        "step_back_context": step_back_context,
        "question": question
    })
    return final_answer
