# Synthetic Data Generation Using RAGAS - RAG Evaluation with LangSmith
# Extracted from Jupyter notebook

# ========================================
# 🤝 BREAKOUT ROOM #1
# ========================================

# Task 1: Dependencies and API Keys
"""
We'll need to install a number of API keys and dependencies, since we'll be leveraging a number of great technologies for this pipeline!

1. OpenAI's endpoints to handle the Synthetic Data Generation
2. OpenAI's Endpoints for our RAG pipeline and LangSmith evaluation
3. QDrant as our vectorstore
4. LangSmith for our evaluation coordinator!

Let's install and provide all the required information below!
"""

# All imports
import nltk
import os
import getpass
from uuid import uuid4
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution, SingleHopSpecificQuerySynthesizer, MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer
from langsmith import Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser
from langsmith.evaluation import LangChainStringEvaluator, evaluate

# NLTK Import
"""
To prevent errors that may occur based on OS - we'll import NLTK and download the needed packages to ensure correct handling of data.
"""
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Environment Setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangChain API Key:")

# Project Name Setup
# We'll also want to set a project name to make things easier for ourselves.
os.environ["LANGCHAIN_PROJECT"] = f"AIM - SDG - {uuid4().hex[0:8]}"

# OpenAI's API Key!

# OpenAI API Key
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# ========================================
# Generating Synthetic Test Data
# ========================================

"""
We will be using Ragas to build out a set of synthetic test questions, references, and reference contexts. This is useful because it will allow us to find out how our system is performing.

NOTE: Ragas is best suited for finding *directional* changes in your LLM-based systems. The absolute scores aren't comparable in a vacuum.
"""

# Data Preparation
"""
We'll prepare our data - which should hopefully be familiar at this point since it's our Loan Data use-case!

Next, let's load our data into a familiar LangChain format using the `DirectoryLoader`.
"""
path = "data/"
loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyMuPDFLoader) # type: ignore
docs = loader.load()

# Knowledge Graph Based Synthetic Generation
"""
Ragas uses a knowledge graph based approach to create data. This is extremely useful as it allows us to create complex queries rather simply. The additional testset complexity allows us to evaluate larger problems more effectively, as systems tend to be very strong on simple evaluation tasks.

Let's start by defining our `generator_llm` (which will generate our questions, summaries, and more), and our `generator_embeddings` which will be useful in building our graph.
"""

# Unrolled SDG
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-nano"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

"""
Next, we're going to instantiate our Knowledge Graph.

This graph will contain N number of nodes that have M number of relationships. These nodes and relationships (AKA "edges") will define our knowledge graph and be used later to construct relevant questions and responses.
"""
kg = KnowledgeGraph()

# Add Documents to Knowledge Graph
"""
The first step we're going to take is to simply insert each of our full documents into the graph. This will provide a base that we can apply transformations to.
"""
### NOTICE: We're using a subset of the data for this example - this is to keep costs/time down.
for doc in docs[:20]:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )

# Apply Transformations
"""
Now, we'll apply the *default* transformations to our knowledge graph. This will take the nodes currently on the graph and transform them based on a set of default transformations.

These default transformations are dependent on the corpus length, in our case:

- Producing Summaries -> produces summaries of the documents
- Extracting Headlines -> finding the overall headline for the document
- Theme Extractor -> extracts broad themes about the documents

It then uses cosine-similarity and heuristics between the embeddings of the above transformations to construct relationships between the nodes.
"""
transformer_llm = generator_llm
embedding_model = generator_embeddings

default_transforms = default_transforms(documents=docs, llm=transformer_llm, embedding_model=embedding_model)
apply_transforms(kg, default_transforms)

# Save and Load Knowledge Graph
# We can save and load our knowledge graphs as follows.
kg.save("loan_data_kg.json")
loan_data_kg = KnowledgeGraph.load("loan_data_kg.json")

# Test Set Generator
# Using our knowledge graph, we can construct a "test set generator" - which will allow us to create queries.
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=loan_data_kg)

# Query Distribution Setup
"""
However, we'd like to be able to define the kinds of queries we're generating - which is made simple by Ragas having pre-created a number of different "QuerySynthesizer"s.

Each of these Synthesizers is going to tackle a separate kind of query which will be generated from a scenario and a persona.

In essence, Ragas will use an LLM to generate a persona of someone who would interact with the data - and then use a scenario to construct a question from that data and persona.

❓ Question #1: What are the three types of query synthesizers doing? Describe each one in simple terms.
"""
query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
]

# Generate Test Set
# Finally, we can use our `TestSetGenerator` to generate our testset!
testset = generator.generate(testset_size=10, query_distribution=query_distribution)

# Abstracted SDG
"""
The above method is the full process - but we can shortcut that using the provided abstractions!

This will generate our knowledge graph under the hood, and will - from there - generate our personas and scenarios to construct our queries.
"""
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs[:20], testset_size=10)

# ========================================
# 🤝 BREAKOUT ROOM #2
# ========================================

# Task 4: LangSmith Dataset
"""
Now we can move on to creating a dataset for LangSmith!

First, we'll need to create a dataset on LangSmith using the `Client`!

We'll name our Dataset to make it easy to work with later.
"""
client = Client()

dataset_name = "Loan Synthetic Data"

langsmith_dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Loan Synthetic Data"
)

# Add Examples to Dataset
for data_row in dataset.to_pandas().iterrows():
  client.create_example(
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

# RAG Setup
rag_documents = docs

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

rag_documents = text_splitter.split_documents(rag_documents)

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Vectorstore
vectorstore = Qdrant.from_documents(
    documents=rag_documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="Loan RAG"
)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# RAG Prompt
RAG_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

# LLM
llm = ChatOpenAI(model="gpt-4.1-mini")

# RAG Chain
rag_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | rag_prompt | llm | StrOutputParser()
)

# Test RAG Chain
rag_chain.invoke({"question" : "What kinds of loans are available?"})

# Evaluation LLM
eval_llm = ChatOpenAI(model="gpt-4.1")

# Evaluators
qa_evaluator = LangChainStringEvaluator("qa", config={"llm" : eval_llm})

labeled_helpfulness_evaluator = LangChainStringEvaluator(
    "labeled_criteria",
    config={
        "criteria": {
            "helpfulness": (
                "Is this submission helpful to the user,"
                " taking into account the correct reference answer?"
            )
        },
        "llm" : eval_llm
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["output"], # type: ignore
        "reference": example.outputs["answer"], # type: ignore
        "input": example.inputs["question"], # type: ignore
    } # type: ignore
)

empathy_evaluator = LangChainStringEvaluator(
    "criteria",
    config={
        "criteria": {
            "empathy": "Is this response empathetic? Does it make the user feel like they are being heard?",
        },
        "llm" : eval_llm
    }
)

# Evaluation
evaluate(
    rag_chain.invoke, # type: ignore
    data=dataset_name,
    evaluators=[
        qa_evaluator, # type: ignore
        labeled_helpfulness_evaluator, # type: ignore
        empathy_evaluator # type: ignore
    ],
    metadata={"revision_id": "default_chain_init"}, # type: ignore
)

# Improved RAG Chain Setup
EMPATHY_RAG_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

You must answer the question using empathy and kindness, and make sure the user feels heard.

Context: {context}
Question: {question}
"""

empathy_rag_prompt = ChatPromptTemplate.from_template(EMPATHY_RAG_PROMPT)

# Updated Documents
rag_documents = docs

# Larger Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 50
)

rag_documents = text_splitter.split_documents(rag_documents)

# Better Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# New Vectorstore
vectorstore = Qdrant.from_documents(
    documents=rag_documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="Loan Data for RAG"
)

# New Retriever
retriever = vectorstore.as_retriever()

# Empathy RAG Chain
empathy_rag_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | empathy_rag_prompt | llm | StrOutputParser()
)

# Test Empathy RAG Chain
empathy_rag_chain.invoke({"question" : "What kinds of loans are available?"})

# Final Evaluation
evaluate(
    empathy_rag_chain.invoke, # type: ignore
    data=dataset_name, # type: ignore
    evaluators=[ # type: ignore
        qa_evaluator, # type: ignore
        labeled_helpfulness_evaluator, # type: ignore
        empathy_evaluator # type: ignore
    ],
    metadata={"revision_id": "empathy_rag_chain"}, # type: ignore
) 