
from dotenv import load_dotenv

load_dotenv()

# Import our custom LLMOps library with production features
from langgraph_agent_lib import (
    ProductionRAGChain,
    CacheBackedEmbeddings, 
    setup_llm_cache,
    create_langgraph_agent,
    create_langgraph_agent_with_helpfulness_check,
    get_openai_model
)

print("✓ LangGraph Agent library imported successfully!")
print("Available components:")
print("  - ProductionRAGChain: Cache-backed RAG with OpenAI")
print("  - LangGraph Agents: Simple and helpfulness-checking agents")
print("  - Production Caching: Embeddings and LLM caching")
print("  - OpenAI Integration: Model utilities")


# Update this path to point to your PDF file
file_path = "./data/The_Direct_Loan_Program.pdf"  # Update this path as needed

# Create a sample document if none exists
import os
if not os.path.exists(file_path):
    print(f"⚠ PDF file not found at {file_path}")
    print("Please update the file_path variable to point to your PDF file")
    print("Or place a PDF file at ./data/sample_document.pdf")
else:
    print(f"✓ PDF file found at {file_path}")

file_path
# Set up production caching for both embeddings and LLM calls
print("Setting up production caching...")

# Set up LLM cache (In-Memory for demo, SQLite for production)
setup_llm_cache(cache_type="memory")
print("✓ LLM cache configured")

# Cache will be automatically set up by our ProductionRAGChain
print("✓ Embedding cache will be configured automatically")
print("✓ All caching systems ready!")

try:
    print("Creating Production RAG Chain...")
    rag_chain = ProductionRAGChain(
        file_path=file_path,
        chunk_size=1000,
        chunk_overlap=100,
        embedding_model="text-embedding-3-small",  # OpenAI embedding model
        llm_model="gpt-4.1-mini",  # OpenAI LLM model
        cache_dir="./cache"
    )
    print("✓ Production RAG Chain created successfully!")
    print(f"  - Embedding model: text-embedding-3-small")
    print(f"  - LLM model: gpt-4.1-mini")
    print(f"  - Cache directory: ./cache")
    print(f"  - Chunk size: 1000 with 100 overlap")
    
except Exception as e:
    print(f"❌ Error creating RAG chain: {e}")
    print("Please ensure the PDF file exists and OpenAI API key is set")


# Create a Simple LangGraph Agent with RAG capabilities
print("Creating LangGraph Agent with helpfulness...")

try:
    agent_with_helpfulness_check = create_langgraph_agent_with_helpfulness_check(
        model_name="gpt-4.1-mini",
        temperature=0.1,
        rag_chain=rag_chain  # Pass our cached RAG chain as a tool
    )
    print("✓ LangGraph Agent with helpfulness created successfully!")
    print("  - Model: gpt-4.1-mini")
    print("  - Tools: Tavily Search, Arxiv, RAG System")
    print("  - Features: Tool calling, parallel execution")
    
except Exception as e:
    print(f"❌ Error creating LangGraph Agent with helpfulness: {e}")
    agent_with_helpfulness_check = None

# Test the Simple Agent
print("🤖 Testing LangGraph Agent with helpfulness...")
print("=" * 50)

test_query = "What are the common repayment timelines for California?"

if agent_with_helpfulness_check:
    try:
        from langchain_core.messages import HumanMessage
        
        # Create message for the agent
        messages = [HumanMessage(content=test_query)]
        
        print(f"Query: {test_query}")
        print("\n🔄 Agent with helpfulness Response:")
        
        # Invoke the agent
        response = agent_with_helpfulness_check.invoke({"messages": messages})
        
        # Extract the final message
        final_message = response["messages"][-1]
        print(final_message.content)
        
        print(f"\n📊 Total messages in conversation: {len(response['messages'])}")
        
    except Exception as e:
        print(f"❌ Error testing agent with helpfulness check: {e}")
else:
    print("⚠ Agent with helpfulness check not available - skipping test")