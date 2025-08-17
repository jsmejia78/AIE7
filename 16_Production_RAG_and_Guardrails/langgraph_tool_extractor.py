"""
LangGraph Tool Extractor

This module provides functionality to extract tools used by a LangGraph agent
from a list of messages, maintaining the chronological order of tool usage.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool


def extract_agent_tools(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """
    Extract tools used by a LangGraph agent from a list of messages.
    
    This function analyzes the message history to identify which tools were used
    and returns them in the order they were first encountered.
    
    Args:
        messages (List[BaseMessage]): List of messages from the LangGraph agent execution
        
    Returns:
        List[Dict[str, Any]]: List of tool information dictionaries, ordered by first use.
                              Each dict contains:
                              - 'name': Tool name
                              - 'description': Tool description (if available)
                              - 'first_used_at': Index of first message where tool was used
                              - 'usage_count': Total number of times the tool was used
                              - 'tool_type': Type of tool (e.g., 'function', 'tool')
    
    Example:
        >>> messages = [AIMessage(content="..."), ToolMessage(tool_name="search", content="...")]
        >>> tools = extract_agent_tools(messages)
        >>> print(tools)
        [{'name': 'search', 'description': None, 'first_used_at': 1, 'usage_count': 1, 'tool_type': 'tool'}]
    """
    if not messages:
        return []
    
    tool_usage = {}
    
    for i, message in enumerate(messages):
        # Check for ToolMessage (when tool returns result)
        if isinstance(message, ToolMessage):
            tool_name = getattr(message, 'tool_name', None) or getattr(message, 'name', None)
            if tool_name:
                if tool_name not in tool_usage:
                    tool_usage[tool_name] = {
                        'name': tool_name,
                        'description': None,
                        'first_used_at': i,
                        'usage_count': 0,
                        'tool_type': 'tool'
                    }
                tool_usage[tool_name]['usage_count'] += 1
        
        # Check for AIMessage with tool calls
        elif isinstance(message, AIMessage):
            # Look for tool_calls in the message
            tool_calls = getattr(message, 'tool_calls', None)
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = getattr(tool_call, 'name', None) or getattr(tool_call, 'tool_name', None)
                    if tool_name:
                        if tool_name not in tool_usage:
                            tool_usage[tool_name] = {
                                'name': tool_name,
                                'description': getattr(tool_call, 'description', None),
                                'first_used_at': i,
                                'usage_count': 0,
                                'tool_type': 'function'
                            }
                        tool_usage[tool_name]['usage_count'] += 1
            
            # Also check for additional_kwargs which might contain tool information
            additional_kwargs = getattr(message, 'additional_kwargs', {})
            if 'tool_calls' in additional_kwargs:
                for tool_call in additional_kwargs['tool_calls']:
                    tool_name = tool_call.get('name') or tool_call.get('tool_name')
                    if tool_name:
                        if tool_name not in tool_usage:
                            tool_usage[tool_name] = {
                                'name': tool_name,
                                'description': tool_call.get('description'),
                                'first_used_at': i,
                                'usage_count': 0,
                                'tool_type': 'function'
                            }
                        tool_usage[tool_name]['usage_count'] += 1
    
    # Sort by first_used_at to maintain chronological order
    sorted_tools = sorted(tool_usage.values(), key=lambda x: x['first_used_at'])
    
    return sorted_tools


def extract_tool_names(messages: List[BaseMessage]) -> List[str]:
    """
    Extract just the names of tools used by a LangGraph agent from a list of messages.
    
    This is a simplified version that returns only the tool names in order of first use.
    
    Args:
        messages (List[BaseMessage]): List of messages from the LangGraph agent execution
        
    Returns:
        List[str]: List of tool names, ordered by first use
        
    Example:
        >>> messages = [AIMessage(content="..."), ToolMessage(tool_name="search", content="...")]
        >>> tool_names = extract_tool_names(messages)
        >>> print(tool_names)
        ['search']
    """
    tools = extract_agent_tools(messages)
    return [tool['name'] for tool in tools]


def get_tool_usage_summary(messages: List[BaseMessage]) -> Dict[str, Any]:
    """
    Get a comprehensive summary of tool usage from LangGraph agent messages.
    
    Args:
        messages (List[BaseMessage]): List of messages from the LangGraph agent execution
        
    Returns:
        Dict[str, Any]: Summary containing:
            - 'total_tools': Number of unique tools used
            - 'total_tool_calls': Total number of tool invocations
            - 'tools_by_usage': Tools sorted by usage frequency (most used first)
            - 'tools_by_order': Tools sorted by first use order
            - 'tool_details': Full tool information from extract_agent_tools()
    """
    tools = extract_agent_tools(messages)
    
    if not tools:
        return {
            'total_tools': 0,
            'total_tool_calls': 0,
            'tools_by_usage': [],
            'tools_by_order': [],
            'tool_details': []
        }
    
    total_tool_calls = sum(tool['usage_count'] for tool in tools)
    
    # Sort by usage count (most used first)
    tools_by_usage = sorted(tools, key=lambda x: x['usage_count'], reverse=True)
    
    # Sort by first use order
    tools_by_order = sorted(tools, key=lambda x: x['first_used_at'])
    
    return {
        'total_tools': len(tools),
        'total_tool_calls': total_tool_calls,
        'tools_by_usage': tools_by_usage,
        'tools_by_order': tools_by_order,
        'tool_details': tools
    }


def test_with_provided_messages():
    """
    Test function using the provided message input to verify tool extraction functionality.
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    # Test messages from the provided input
    test_messages = [
        HumanMessage(
            content='What is the main purpose of the Direct Loan Program?',
            additional_kwargs={},
            response_metadata={},
            id='8633a9c6-d7ab-4d61-b046-2aec9371f1c4'
        ),
        AIMessage(
            content='',
            additional_kwargs={
                'tool_calls': [{
                    'id': 'call_ncL3NIqBVBJwRvAkbElxYXKI',
                    'function': {
                        'arguments': '{"query":"main purpose of the Direct Loan Program"}',
                        'name': 'retrieve_information'
                    },
                    'type': 'function'
                }],
                'refusal': None
            },
            response_metadata={
                'token_usage': {
                    'completion_tokens': 20,
                    'prompt_tokens': 191,
                    'total_tokens': 211
                },
                'model_name': 'gpt-4.1-mini-2025-04-14',
                'finish_reason': 'tool_calls'
            },
            id='run--7a0bac16-0ba7-460d-a04b-55190c81bb9d-0',
            tool_calls=[{
                'name': 'retrieve_information',
                'args': {'query': 'main purpose of the Direct Loan Program'},
                'id': 'call_ncL3NIqBVBJwRvAkbElxYXKI',
                'type': 'tool_call'
            }],
            usage_metadata={'input_tokens': 191, 'output_tokens': 20, 'total_tokens': 211}
        ),
        ToolMessage(
            content='The main purpose of the Direct Loan Program, under the William D. Ford Federal Direct Loan Program, is for the U.S. Department of Education to make loans to help students and parents pay the cost of attendance (COA) at a postsecondary school.',
            name='retrieve_information',
            id='ff4f4733-4060-418b-8db4-7676637906a8',
            tool_call_id='call_ncL3NIqBVBJwRvAkbElxYXKI'
        ),
        AIMessage(
            content='The main purpose of the Direct Loan Program is for the U.S. Department of Education to provide loans to help students and parents pay the cost of attendance at a postsecondary school.',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 37,
                    'prompt_tokens': 271,
                    'total_tokens': 308
                },
                'model_name': 'gpt-4.1-mini-2025-04-14',
                'finish_reason': 'stop'
            },
            id='run--5458ebad-0aab-4d59-bc93-b2bc1daf8d18-0',
            usage_metadata={'input_tokens': 271, 'output_tokens': 37, 'total_tokens': 308}
        )
    ]
    
    print("=" * 50)
    print("TESTING WITH PROVIDED MESSAGE INPUT")
    print("=" * 50)
    
    # Test 1: extract_agent_tools
    print("\n1. Testing extract_agent_tools():")
    tools = extract_agent_tools(test_messages)
    print(f"   Extracted tools: {tools}")
    
    # Test 2: extract_tool_names
    print("\n2. Testing extract_tool_names():")
    tool_names = extract_tool_names(test_messages)
    print(f"   Tool names in order: {tool_names}")
    
    # Test 3: get_tool_usage_summary
    print("\n3. Testing get_tool_usage_summary():")
    summary = get_tool_usage_summary(test_messages)
    print(f"   Summary: {summary}")
    
    # Test 4: Verify expected results
    print("\n4. Verification:")
    expected_tool = 'retrieve_information'
    if tool_names and tool_names[0] == expected_tool:
        print(f"   ✓ First tool is '{expected_tool}' as expected")
    else:
        print(f"   ✗ Expected first tool '{expected_tool}', got {tool_names[0] if tool_names else 'None'}")
    
    if len(tool_names) == 1:
        print(f"   ✓ Found exactly 1 unique tool as expected")
    else:
        print(f"   ✗ Expected 1 unique tool, found {len(tool_names)}")
    
    if tools and tools[0]['usage_count'] == 1:
        print(f"   ✓ Tool used exactly 1 time as expected")
    else:
        print(f"   ✗ Expected tool to be used 1 time, got {tools[0]['usage_count'] if tools else 'None'}")
    
    print("\n" + "=" * 50)
    return tools, tool_names, summary


def test_edge_cases():
    """
    Test function for edge cases and error handling.
    """
    print("\nTESTING EDGE CASES")
    print("=" * 50)
    
    # Test empty messages
    print("\n1. Testing with empty messages:")
    empty_result = extract_agent_tools([])
    print(f"   Empty messages result: {empty_result}")
    
    # Test None messages
    print("\n2. Testing with None messages:")
    try:
        none_result = extract_agent_tools(None)
        print(f"   None messages result: {none_result}")
    except Exception as e:
        print(f"   None messages error: {e}")
    
    # Test messages without tools
    print("\n3. Testing with messages without tools:")
    from langchain_core.messages import HumanMessage
    no_tool_messages = [HumanMessage(content="Hello, how are you?")]
    no_tool_result = extract_agent_tools(no_tool_messages)
    print(f"   No tool messages result: {no_tool_result}")
    
    print("=" * 50)


def run_all_tests():
    """
    Run all test functions to verify the tool extraction functionality.
    """
    print("RUNNING ALL TESTS FOR LANGGRAPH TOOL EXTRACTOR")
    print("=" * 60)
    
    try:
        # Test with provided messages
        tools, tool_names, summary = test_with_provided_messages()
        
        # Test edge cases
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests when file is executed directly
    run_all_tests()


