import getpass
import os
from dataloader import load_data
from uuid import uuid4

use_api_keys_input = True # Set to True to use API keys from input

if use_api_keys_input:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key:")
    os.environ["COHERE_API_KEY"] = getpass.getpass("Cohere API Key:")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangChain API Key:")
    os.environ["LANGCHAIN_PROJECT"] = f"AIM - ADVANCED RETRIEVAL - {uuid4().hex[0:8]}"

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from retrievers import get_retrieval_chains_and_wrappers
from rag_prompt import get_rag_prompt
from langchain_qdrant import Qdrant
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.stores import InMemoryStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas import EvaluationDataset
from ragas import evaluate as evaluate_ragas, RunConfig
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ContextEntityRecall,
)
from langsmith.evaluation import LangChainStringEvaluator, evaluate as evaluate_langsmith
from langsmith import Client as ClientLangSmith
import numpy as np

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

client_qdrant = QdrantClient(location=":memory:")
client_qdrant.create_collection(
    collection_name="full_documents",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
)

parent_document_vectorstore = QdrantVectorStore(
    collection_name="full_documents", embedding=OpenAIEmbeddings(model="text-embedding-3-small"), client=client_qdrant
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
chains, wrappers = get_retrieval_chains_and_wrappers(retrievers_config, 
                            loan_complaint_data, 
                            rag_prompt, 
                            chat_model, 
                            MODE)

eval_full_results = {}
eval_summary_results = {}
eval_langsmith_raw_results = {}
eval_langsmith_summary_results ={}

for chain_name in chains.keys():

    golden_dataset_active_copy = golden_dataset.copy()

    for test_row in golden_dataset_active_copy:
        response = wrappers[chain_name](test_row.eval_sample.user_input)
        test_row.eval_sample.response = response["response"]
        test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]

    evaluation_active_dataset = EvaluationDataset.from_pandas(golden_dataset_active_copy.to_pandas())
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))

    custom_run_config = RunConfig(timeout=360)

    # Run evaluation using only RETRIEVAL metrics
    eval_result_active = evaluate_ragas(
        dataset=evaluation_active_dataset,
        metrics=[LLMContextPrecisionWithReference(),LLMContextRecall(), ContextEntityRecall()],
        llm=evaluator_llm,
        run_config=custom_run_config
    )

    eval_full_results[chain_name] = eval_result_active

    # Convert to DataFrames
    df_eval_result_active = eval_result_active.to_pandas()

    # Compute means and standard deviations
    eval_result_active_means = df_eval_result_active.mean(numeric_only=True)
    eval_result_active_stds = df_eval_result_active.std(numeric_only=True)

    eval_summary_results[chain_name] = {
        "means": eval_result_active_means,
        "stds": eval_result_active_stds
    }   

    # ===============================
    # Eval with LangSmith
    # ===============================

    client_langsmith = ClientLangSmith()

    dataset_name = f"Advanced Retrieval - {chain_name}"

    langsmith_dataset = client_langsmith.create_dataset(
        dataset_name=dataset_name,
        description=f"Advanced Retrieval - {chain_name}"
    )

    for data_row in evaluation_active_dataset.to_pandas().iterrows():
        client_langsmith.create_example(
            inputs={
                "question": data_row[1]["user_input"]
            },
            outputs={
                "answer": data_row[1]["reference"]
            },
            metadata={
                "context": data_row[1]["reference_contexts"]
            },
            dataset_id=langsmith_dataset.id
        )

    eval_llm_langsmith = ChatOpenAI(model="gpt-4.1-mini") # it was 4.1 full

    qa_evaluator = LangChainStringEvaluator("qa", config={"llm" : eval_llm_langsmith})

    context_relevance_evaluator = LangChainStringEvaluator(
        "labeled_criteria",
        config={
            "criteria": {
                "context_relevance": (
                    "How relevant is the retrieved context to the input question?"
                    " Rate it based on whether the context helps answer the question directly,"
                    " contains distracting/unrelated info, or is missing key facts."
                )
            },
            "llm": eval_llm_langsmith,
        },
        prepare_data=lambda run, example: {
            "prediction": run.inputs.get("context", ""),   # retrieved context (string or joined list)
            "reference": example.outputs.get("answer", ""),  # optional: reference answer
            "input": example.inputs["question"],           # the query
        },
    )

    eval_langsmith_result_active = evaluate_langsmith(
        chains[chain_name].invoke, # type: ignore
        data=evaluation_active_dataset,
        evaluators=[
            qa_evaluator, # type: ignore
            context_relevance_evaluator, # type: ignore
        ],
        metadata={"revision_id": "default_chain_init"},
    )
    
    all_latencies = [res.run.execution_time for res in eval_langsmith_result_active]
    all_costs = [res.run.cost for res in eval_langsmith_result_active]

    average_latency = np.mean(all_latencies)
    total_cost = sum(all_costs)

    eval_langsmith_raw_results[chain_name] = eval_langsmith_result_active

    eval_langsmith_summary_results[chain_name] = {
        "average_latency": average_latency,
        "total_cost": total_cost
    }



