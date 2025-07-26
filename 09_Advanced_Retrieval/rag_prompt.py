from langchain.prompts import ChatPromptTemplate

def get_rag_prompt():
    # Create the prompt template for all the retrieval methods
    RAG_TEMPLATE = """\
    You are a helpful and kind assistant. Use the context provided below to answer the question.

    If you do not know the answer, or are unsure, say you don't know.

    Query:
    {question}

    Context:
    {context}
    """

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    return rag_prompt