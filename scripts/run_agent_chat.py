#!/usr/bin/env python3
"""
Run the agent chatbot from the command line.

Usage:
    python scripts/run_agent_chat.py "When did the Roman Republic end?"
    python scripts/run_agent_chat.py "How does photosynthesis work?"

Prerequisites:
    1. LanceDB indexes built (vector + keyword FTS)
    2. Local model server running: bash scripts/start_model_server.sh
"""

import sys

from workbench.systems.chatbot import ChatbotSystem


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_agent_chat.py "<your question>"')
        raise SystemExit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")

    system = ChatbotSystem()
    state = system.ask(question)

    # Print results
    print("\n" + "=" * 60)

    if state.error:
        print(f"ERROR: {state.error}")
    else:
        print(f"Answer:\n{state.answer_text}")

    print("\n--- Metadata ---")
    print(f"  Steps:          {state.step}")
    print(f"  Retrieved:      {len(state.retrieved)} chunks")
    print(f"  Opened:         {len(state.opened)} chunks")
    print(f"  Citations:      {state.citations}")
    print(f"  Tokens in/out:  {state.tokens_in} / {state.tokens_out}")
    print(f"  Model latency:  {state.model_latency_ms:.0f} ms")

    if state.opened:
        print("\n--- Evidence used ---")
        for chunk in state.opened:
            print(f"  [{chunk.chunk_id}] {chunk.title}")


if __name__ == "__main__":
    main()
