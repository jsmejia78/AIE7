from langgraph_sdk import get_sync_client

def main():
    client = get_sync_client(url="http://localhost:2024")
    for chunk in client.runs.stream(
        None,  # Threadless run
        "agent_with_helpfulness",  # Assistant id from langgraph.json (assistants)
        input={
            "messages": [
                {
                    "role": "human",
                    "content": "Where should I invest GOOG or AAPL such that I can improve my changes of repayment for my student loan?",
                }
            ]
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

if __name__ == "__main__":
    main()