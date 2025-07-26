import getpass
import os
from dataloader import load_data

use_api_keys_input = False # Set to True to use API keys from input

if use_api_keys_input:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key:")
    os.environ["COHERE_API_KEY"] = getpass.getpass("Cohere API Key:")

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from retrievers import get_retrieval_chains_and_wrapers
from rag_prompt import get_rag_prompt
from langchain_qdrant import Qdrant
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.stores import InMemoryStore
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import QdrantClient
from langchain_qdrant import models
from langchain_experimental.text_splitter import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

MODE = "naive-baseline"  # semantic

# ===============================
# Load the data
# ===============================
loan_complaint_data = load_data(num_docs=50)

# ===============================
# Create the embeddings, chat model, and prompt template
# ===============================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chat_model = ChatOpenAI(model="gpt-4.1-nano")
rag_prompt = get_rag_prompt()

# ===============================
# update the data to semantic chunking if MODE is "semantic"
# ===============================
if MODE == "naive-baseline":
    # ===============================
    # Naive Retrieval
    # ===============================

    vectorstore = Qdrant.from_documents(
        loan_complaint_data,
        embeddings,
        location=":memory:",
        collection_name="LoanComplaints"
    )
    
elif MODE == "semantic":
    # ===============================
    # Semantic Retrieval
    # ===============================

    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    loan_complaint_data = semantic_chunker.split_documents(loan_complaint_data)
    semantic_vectorstore = Qdrant.from_documents(
        loan_complaint_data,
        embeddings,
        location=":memory:",
        collection_name="Loan_Complaint_Data_Semantic_Chunks"
    )
    vectorstore = semantic_vectorstore
else:
    raise ValueError(f"Invalid mode: {MODE}")

# ===============================
# Create RAGAS datataset
# ===============================

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
golden_dataset = generator.generate_with_langchain_docs(loan_complaint_data, testset_size=50)

#dataset.to_pandas()

# ===============================
# Parent Document Retrieval
# ===============================
# Create the retriever - parent document retrieval
parent_docs = loan_complaint_data
child_splitter = RecursiveCharacterTextSplitter(chunk_size=750)

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="full_documents",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
)

parent_document_vectorstore = QdrantVectorStore(
    collection_name="full_documents", embedding=OpenAIEmbeddings(model="text-embedding-3-small"), client=client
)

in_memory_store = InMemoryStore()

retrievers_config = {
    "naive": {
        "vectorstore": vectorstore},
    "parent_document": {
        "vectorstore": parent_document_vectorstore,
        "in_memory_store": in_memory_store,
        "child_splitter": child_splitter
    }
}

# ===============================
# Create RAG retrievers and chains
# ===============================
retrieval_chains_and_wrapers = get_retrieval_chains_and_wrapers(retrievers_config, 
                            loan_complaint_data, 
                            rag_prompt, 
                            chat_model, 
                            MODE)


