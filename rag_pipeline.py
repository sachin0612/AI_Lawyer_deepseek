from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load the LLM model via Groq
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")

custom_prompt_template = """
You are an AI lawyer. Use the context below to answer the user's question.
If the answer is not in the context, say "I don't know" and do not make up an answer.

Question: {question}
Context: {context}
Answer:
"""

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

def answer_query(documents, query):
    context = get_context(documents)
    prompt = PromptTemplate.from_template(custom_prompt_template)
    formatted_prompt = prompt.format(question=query, context=context)
    return llm_model.invoke(formatted_prompt)
