#!/usr/bin/env python3
"""
Run the agent chatbot from the command line.

Usage:
    python scripts/run_agent_chat.py "When did the Roman Republic end?"
    python scripts/run_agent_chat.py --graph researcher_graph "Timeline of the fall of Rome"

Prerequisites:
    1. LanceDB indexes built (vector + keyword FTS)
    2. Local model server running: bash scripts/start_model_server.sh
"""

import argparse

from workbench.systems.runner import GraphRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent chatbot")
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument(
        "--graph",
        default="rag_graph",
        choices=["rag_graph", "researcher_graph", "supervisor_graph"],
        help="Which graph to run (default: rag_graph)",
    )
    args = parser.parse_args()

    question = " ".join(args.question)
    graph_name = args.graph

    print(f"Question: {question}")
    print(f"Graph:    {graph_name}\n")

    runner = GraphRunner()
    result = runner.run(graph_name, question)

    # Print results
    print("\n" + "=" * 60)

    if result.get("error"):
        print(f"ERROR: {result['error']}")
    else:
        print(f"Answer:\n{result.get('answer_text', '(no answer)')}")

    print("\n--- Metadata ---")
    print(f"  Retrieved:      {len(result.get('retrieved', []))} chunks")
    print(f"  Opened:         {len(result.get('opened', []))} chunks")
    print(f"  Citations:      {result.get('citations', [])}")
    print(
        f"  Tokens in/out:  {result.get('tokens_in', 0)} / {result.get('tokens_out', 0)}"
    )
    print(f"  Model latency:  {result.get('model_latency_ms', 0):.0f} ms")

    if result.get("timeline"):
        print(f"\n--- Timeline ({len(result['timeline'])} events) ---")
        for item in result["timeline"]:
            date = item.get("date", "?")
            event = item.get("event", "?")
            print(f"  {date}: {event}")

    if result.get("opened"):
        print("\n--- Evidence used ---")
        for chunk in result["opened"]:
            cid = chunk.get("chunk_id", "?")
            title = chunk.get("title", "?")
            print(f"  [{cid}] {title}")


if __name__ == "__main__":
    main()
