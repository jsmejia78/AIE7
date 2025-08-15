# mcp_react_agent_simple.py
import os
import asyncio
import logging
from typing import Dict, Any, List, Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import httpx
from uuid import uuid4
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)

HELPFUL_ANSWER_CHECK_TEMPLATE = """\
"You are a concise assistant that can reach to another agent (server) using A2A protocol and get the response.
Query to Server Agent :
{query}
Answer from Server Agent :
{server_response}
If the answer is helpful (answers the query), return "FINISH" only.
If the answer is not helpful (does not properly answer the query), return "CONTINUE" only, to try again.
"""

REMOTE_AGENT_SERVER_REACH_TEMPLATE = """\
Query:
{query}
Previous remote agent responses (if any):
{previous_server_responses}
"""

load_dotenv()

class AgentState(TypedDict):
    """ Agent state for the agent graph. """
    messages: Annotated[List, add_messages]
    query: str
    server_response: str
    previous_server_responses: List[str]

class ClientAgentForA2AServer:
    def __init__(
        self,
        server_url: str = "http://localhost:10000",
        openai_model: str = None,
        temperature: float = 0.0,
    ):

        self.server_url = server_url

        # Initialize these upfront
        self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self.agent_card = None  # Will be populated in _initialize_agent_card

        # LLM mode, graph, helpful check prompt
        self.model_name = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(model=self.model_name, temperature=temperature)
        self._init_graph()
        self.helpful_answer_check_prompt = ChatPromptTemplate.from_template(HELPFUL_ANSWER_CHECK_TEMPLATE)

        self.helpful_chain = (
            {"query": itemgetter("query"), "server_response": itemgetter("server_response")}
            | self.helpful_answer_check_prompt
            | self.llm
            | StrOutputParser()
        )
        self.final_state = None
        self.query = ""
        self.previous_answers = []
        
        # Initialize the agent card - will be awaited when needed
        self._agent_card_initialized = False

    async def _initialize_agent_card(self):
        """Initialize the agent card once during startup."""
        if self._agent_card_initialized:
            return
            
        # Configure logging to show INFO level messages
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Fetch Public Agent Card and Initialize Client
        final_agent_card_to_use: AgentCard | None = None

        try:
            resolver = A2ACardResolver(
                httpx_client=self.httpx_client,
                base_url=self.server_url,
            )
            
            logger.info(
                f'Attempting to fetch public agent card from: {self.server_url}{AGENT_CARD_WELL_KNOWN_PATH}'
            )
            _public_card = (
                await resolver.get_agent_card()
            )  # Fetches from default public path
            logger.info('Successfully fetched public agent card:')
            logger.info(
                _public_card.model_dump_json(indent=2, exclude_none=True)
            )
            final_agent_card_to_use = _public_card
            logger.info(
                '\nUsing PUBLIC agent card for client initialization (default).'
            )

            if _public_card.supports_authenticated_extended_card:
                try:
                    logger.info(
                        '\nPublic card supports authenticated extended card. '
                        'Attempting to fetch from: '
                        f'{self.server_url}{EXTENDED_AGENT_CARD_PATH}'
                    )
                    auth_headers_dict = {
                        'Authorization': 'Bearer dummy-token-for-extended-card'
                    }
                    _extended_card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={'headers': auth_headers_dict},
                    )
                    logger.info(
                        'Successfully fetched authenticated extended agent card:'
                    )
                    logger.info(
                        _extended_card.model_dump_json(
                            indent=2, exclude_none=True
                        )
                    )
                    final_agent_card_to_use = (
                        _extended_card  # Update to use the extended card
                    )
                    logger.info(
                        '\nUsing AUTHENTICATED EXTENDED agent card for client '
                        'initialization.'
                    )
                except Exception as e_extended:
                    logger.warning(
                        f'Failed to fetch extended agent card: {e_extended}. '
                        'Will proceed with public card.',
                        exc_info=True,
                    )
            elif (
                _public_card
            ):  # supports_authenticated_extended_card is False or None
                logger.info(
                    '\nPublic card does not indicate support for an extended card. Using public card.'
                )
            
            # Set the final agent card
            self.agent_card = final_agent_card_to_use
            self._agent_card_initialized = True
            logger.info('Agent card initialization completed successfully.')
                    
        except Exception as e:
            logger.error(f'Failed to initialize agent card: {e}', exc_info=True)
            raise

    async def _reach_to_server(self, state: AgentState) -> AgentState:
        """ Reach to the server and get the response. """
        try:
            my_prev = state.get("previous_server_responses", [])
            my_prev_str = "\n".join(my_prev) if my_prev else "None"
            query_to_send = REMOTE_AGENT_SERVER_REACH_TEMPLATE.format(
                query= state.get("query", ""),
                previous_server_responses=my_prev_str
            )

            # Configure logging to show INFO level messages
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)  # Get a logger instance

            # Ensure agent card is initialized
            if not self._agent_card_initialized:
                await self._initialize_agent_card()

            # Create client with existing httpx client and agent card
            client = A2AClient(
                httpx_client=self.httpx_client, 
                agent_card=self.agent_card
            )
            logger.info('A2AClient initialized.')

            send_message_payload: dict[str, Any] = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': f'{query_to_send}'}
                    ],
                    'message_id': uuid4().hex,
                },
            }
            request = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload)
            )

            response = await client.send_message(request)
            #print(response.model_dump(mode='json', exclude_none=True))

            return {"messages": [("ai", response.content)],
                "server_response": response.content
            }
        except Exception as e:
            error_msg = f"Failed to reach server: {str(e)}"
            logging.error(error_msg)
            return {"messages": [("ai", error_msg)],
                "server_response": error_msg
            }

    def _helpful_answer_check_node(self, state: AgentState) -> AgentState:
        """ Check the helpfulness of the response. """
        try:
            answer_text = self.helpful_chain.invoke(state)
            return {"messages": [("ai", answer_text)],
                "previous_server_responses": state.get("previous_server_responses", []) + [state.get("server_response", "")]
            }
        except Exception as e:
            error_msg = f"Failed to check answer helpfulness: {str(e)}"
            logging.error(error_msg)
            return {"messages": [("ai", "FINISH")],  # Default to finish on error
                "previous_server_responses": state.get("previous_server_responses", []) + [state.get("server_response", "")]
            }

    def _should_continue(self, state: AgentState) -> AgentState:
        """ Check if the agent should continue. """
        last_message = state["messages"][-1]
        if last_message.content == "CONTINUE":
            self.previous_answers.append(last_message.content)
            return "reach_agent_server"
        elif last_message.content == "FINISH":
            return END
        else:
            # If the last message is not "CONTINUE" or "FINISH", then continue to retry server call and retry helpful check
            self.previous_answers.append(last_message.content)
            return "reach_agent_server"

    def _init_graph(self):
        """ Initialize the graph. """
        graph = StateGraph(AgentState)
        graph.add_node("reach_agent_server", self._reach_to_server)
        graph.add_node("helpful_answer_check", self._helpful_answer_check_node)
        graph.add_edge("reach_agent_server", "helpful_answer_check")
        graph.add_conditional_edges(
            "helpful_answer_check",
            self._should_continue,
            {
                "CONTINUE": "reach_agent_server",
                "FINISH": END,
            },
        )
        graph.set_entry_point("reach_agent_server")
        self.graph = graph.compile()

    async def chat(self, message: str) -> str:
        """Chat with the agent asynchronously."""
        try:
            if not message or not message.strip():
                return "Please provide a valid message."
                
            self.query = message
            self.final_state = await self.graph.ainvoke({"query": message})
            
            # Extract the server response from the final state
            if "server_response" in self.final_state:
                return self.final_state["server_response"]
            elif "messages" in self.final_state and self.final_state["messages"]:
                last_message = self.final_state["messages"][-1]
                if isinstance(last_message, tuple):
                    return last_message[1]  # Return content from tuple format
                elif hasattr(last_message, 'content'):
                    return last_message.content
            
            return "No response available"
        except Exception as e:
            error_msg = f"Error during chat: {str(e)}"
            logging.error(error_msg)
            return error_msg
    
    def get_final_state(self) -> Dict[str, Any]:
        return self.final_state
    
    async def cleanup(self):
        """Clean up resources, especially the httpx client."""
        try:
            if hasattr(self, 'httpx_client') and not self.httpx_client.is_closed:
                await self.httpx_client.aclose()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup of async resources."""
        if hasattr(self, 'httpx_client') and not self.httpx_client.is_closed:
            # Note: This is not ideal for async cleanup, but it's a fallback
            # The user should call cleanup() explicitly when done
            pass

# Example usage
if __name__ == "__main__":
    async def main():
        agent = ClientAgentForA2AServer()
        try:
            print("AI Agent Chat - Type 'quit' to exit")
            print("-" * 40)
            
            while True:
                # Get user input
                user_input = input("How can I help you today?: ").strip()
                
                # Check if user wants to quit
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                try:
                    # Get response from agent
                    print("Agent is thinking...")
                    response = await agent.chat(user_input)
                    print(f"\nAgent: {response}")
                except Exception as e:
                    print(f"\nError: {e}")
                    print("Please try again.")
                    
        finally:
            # Clean up resources
            await agent.cleanup()
    
    # Run the async main function
    asyncio.run(main())
    