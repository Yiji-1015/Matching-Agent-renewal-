import argparse

import pandas as pd
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from matching_agent import build_matching_graph, get_initial_state
from matching_agent.config import DEFAULT_DATASET_PATH, DEFAULT_RECURSION_LIMIT


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Matching Agent pipeline.")
    parser.add_argument("--message", help="Input message to match.")
    parser.add_argument("--username", default="Guest", help="Username to exclude from candidates.")
    parser.add_argument("--row", type=int, help="Use a row from 0522_data.xlsx instead of --message.")
    parser.add_argument("--thread-id", default="demo", help="LangGraph checkpoint thread id.")
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print the node-level execution trace after the final result.",
    )
    return parser.parse_args()


def load_input_from_row(row_index: int):
    df = pd.read_excel(DEFAULT_DATASET_PATH)
    row = df.iloc[row_index]
    return row["Message"], row["User"]


def main():
    load_dotenv()
    args = parse_args()

    if args.row is not None:
        input_message, username = load_input_from_row(args.row)
    elif args.message:
        input_message, username = args.message, args.username
    else:
        raise SystemExit("Use either --message or --row.")

    graph = build_matching_graph()
    state = get_initial_state(input_message, username)
    config = RunnableConfig(
        recursion_limit=DEFAULT_RECURSION_LIMIT,
        configurable={"thread_id": args.thread_id},
    )
    result = graph.invoke(state, config=config)

    print("\n=== Input ===")
    print(result["input_message"])
    print("\n=== Reformed Query ===")
    print(result.get("reformed_query"))
    print("\n=== Matched User ===")
    print(result.get("matched_username"))
    print("\n=== Matched Message ===")
    print(result.get("matched_message"))
    print("\n=== Certainty ===")
    print(result.get("certainty"))
    print("\n=== Last Agent ===")
    print(result.get("last_agent"))
    print("\n=== Evaluations / Retries ===")
    print(result.get("evaluation_count"), "/", result.get("retry_count"))

    if args.show_trace:
        print("\n=== LangGraph Trace ===")
        for event in result.get("trace", []):
            print(f"[{event['node']}] {event['detail']}")


if __name__ == "__main__":
    main()
