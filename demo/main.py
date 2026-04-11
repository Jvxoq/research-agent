import argparse
from langchain_core.messages import HumanMessage
from agent.graph import graph


def main() -> None:
    """Run the research agent from the command line."""
    parser = argparse.ArgumentParser(description="Run the LangGraph research agent")
    parser.add_argument("query", help="Research question")
    args = parser.parse_args()

    state = {
        "messages": [HumanMessage(content=args.query)],
        "initial_search_query_count": 2,
        "max_research_loops": 1,
        "reasoning_model": "",
    }
    try:
        result = graph.invoke(state)
        messages = result.get("messages", [])
        if messages:
            print(f"\n---------- <Resarch Report> ----------\n{messages[-1].content}\n")
        else:
            print("Please try again after a while.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please try again after a while.")


if __name__ == "__main__":
    main()
