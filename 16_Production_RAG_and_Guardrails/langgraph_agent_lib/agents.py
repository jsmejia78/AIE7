"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate
from guardrails import Guard
from .models import get_openai_model
from .rag import ProductionRAGChain
from langchain_core.messages import SystemMessage, HumanMessage
from .guards import run_all_guardrails_parallel

class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]

def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools

#==============================================
# Simple Agent
#==============================================


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a simple LangGraph agent.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")
    
    return graph.compile()

#==============================================
# Agent with Helpfulness Check
#==============================================

def create_langgraph_agent_with_helpfulness_check(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a LangGraph agent with a helpfulness check.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    
    def helpfulness_node(state: AgentState) -> Dict[str, Any]: #NOTE: I added this node to the graph.
        """Evaluate helpfulness of the latest response relative to the initial query."""
        # If we've exceeded loop limit, short-circuit with END decision marker
        if len(state["messages"]) > 10:
            return {"messages": [AIMessage(content="HELPFULNESS:END")]}    

        initial_query = state["messages"][0]
        final_response = state["messages"][-1]

        prompt_template = """
        Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

        Initial Query:
        {initial_query}

        Final Response:
        {final_response}"""

        helpfulness_prompt_template = PromptTemplate.from_template(prompt_template)
        helpfulness_check_model = get_openai_model(model_name="gpt-4.1-mini")
        helpfulness_chain = (
            helpfulness_prompt_template | helpfulness_check_model | StrOutputParser()
        )

        helpfulness_response = helpfulness_chain.invoke(
            {
                "initial_query": initial_query.content,
                "final_response": final_response.content,
            }
        )

        decision = "Y" if "Y" in helpfulness_response else "N"
        return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}

    
    def helpfulness_decision(state: AgentState):
        """Terminate on 'HELPFULNESS:Y' or loop otherwise; guard against infinite loops."""
        # Check loop-limit marker
        if any(getattr(m, "content", "") == "HELPFULNESS:END" for m in state["messages"][-1:]):
            return END

        last = state["messages"][-1]
        text = getattr(last, "content", "")
        if "HELPFULNESS:Y" in text:
            return "end"
        return "continue"

    def route_to_action_or_helpfulness(state: AgentState):
        """Decide whether to execute tools or run the helpfulness evaluator."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "helpfulness"
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", helpfulness_node)
    graph.set_entry_point("agent")

    graph.add_edge("action", "agent")

    graph.add_conditional_edges(
        "agent",
        route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"},
    )
    graph.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"continue": "agent", "end": END, END: END},
    )
    
    return graph.compile()

#==============================================
# Agent with Guardrails
#==============================================

class AgentStateWithGuardrails(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]
    query_sanitized: str
    input_guardrail_status: str # blocked, redacted, or allowed   
    output_guardrail_status: str # blocked, redacted, or allowed   


def create_langgraph_agent_with_guardrails(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    guardrails_input: Optional[Dict[str, Any]] = None,
    guardrails_output: Optional[Dict[str, Any]] = None
):
    """Create a simple LangGraph agent.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        guardrails: Optional dictionary of guardrails to apply to the agent
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentStateWithGuardrails) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentStateWithGuardrails):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "output_guards"
    
    async def input_guards(state: AgentStateWithGuardrails) -> Dict[str, Any]:
        """Guardrails for the input messages."""

        user_input = state["messages"][-1].content

        guardrail_results = await run_all_guardrails_parallel(guardrails_input, user_input)
        updates = {}
        updates["input_guardrail_status"] = "passed"

        for guardrail_name, guardrail_result in guardrail_results.items():
            
            if guardrail_result["action"] == "blocked":
                updates = {"input_guardrail_status": "blocked"}
                updates["messages"] = [
                    SystemMessage(content=f"User input blocked for {guardrail_name}. Please try again."),
                ]
                return updates

            elif guardrail_result["action"] == "redacted":
                updates["messages"] = [
                    SystemMessage(content=f"Note: user input sanitized for {guardrail_name}."),
                    HumanMessage(content=guardrail_result["sanitized_text"] )
                ]
                updates["input_guardrail_status"]= "redacted"

        return updates

    async def output_guards(state: AgentStateWithGuardrails) -> Dict[str, Any]:
        """Guardrails for the output messages."""

        agent_response = state["messages"][-1].content

        guardrail_results = await run_all_guardrails_parallel(guardrails_output, agent_response)
        updates = {}

        for guardrail_name, guardrail_result in guardrail_results.items():
            
            if guardrail_result["action"] == "blocked":
                updates = {"output_guardrail_status": "blocked"}
                updates["messages"] = [
                    SystemMessage(content=f"LLM output blocked for {guardrail_name}. LLM lease try again with an updated query."),
                ]
                return updates

            elif guardrail_result["action"] == "redacted":
                updates["messages"] = [
                    SystemMessage(content=f"Note: LLM output sanitized for {guardrail_name}."),
                    AIMessage(content=guardrail_result["sanitized_text"] )
                ]
                updates = {"output_guardrail_status": "redacted"}

        # If we get here, all guardrails passed
        updates["output_guardrail_status"] = "passed"
        return updates

    def should_continue_after_input_guards(state: AgentStateWithGuardrails):
        """Guardrails for the input messages."""
        guardrail_status = state["input_guardrail_status"]
        if guardrail_status == "blocked":
            return END
        elif guardrail_status == "redacted":
            return "agent"
        elif guardrail_status == "passed":
            return "agent"
        else:
            return "agent"

    def should_continue_after_output_guards(state: AgentStateWithGuardrails):
        """Guardrails for the input messages."""
        guardrail_status = state["output_guardrail_status"]
        if guardrail_status == "blocked":
            return "agent"
        elif guardrail_status == "redacted":
            return END
        elif guardrail_status == "passed":
            return END
        else:
            return END

    # Build graph
    graph = StateGraph(AgentStateWithGuardrails)
    tool_node = ToolNode(tools)
    
    graph.add_node("input_guards", input_guards)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("output_guards", output_guards)
    graph.set_entry_point("input_guards")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", "output_guards": "output_guards"})
    graph.add_conditional_edges("input_guards", should_continue_after_input_guards, {"agent": "agent", END: END})
    graph.add_conditional_edges("output_guards", should_continue_after_output_guards, {"agent": "agent", END: END})
    graph.add_edge("action", "agent")
       
    return graph.compile()

