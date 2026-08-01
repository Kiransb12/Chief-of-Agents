"""
Minimal terminal client for testing the /chat endpoint with Memory Consolidation on exit.

Usage:
    python scripts/chat_cli.py
"""
import uuid
import requests

API_URL = "http://127.0.0.1:8000/chat"
CONSOLIDATE_URL = "http://127.0.0.1:8000/session/consolidate"


def main() -> None:
    session_id = str(uuid.uuid4())
    print("Chief of Staff — Phase 3 CLI (Memory). Type 'exit' to quit.\n")
    while True:
        try:
            message = input("You: ").strip()
        except KeyboardInterrupt:
            break
        if message.lower() in ("exit", "quit"):
            break
        if not message:
            continue
        try:
            resp = requests.post(API_URL, json={"session_id": session_id, "message": message})
            resp.raise_for_status()
            data = resp.json()
            print(f"\nAssistant [{data['route']}, rag={data['used_rag']}]: {data['reply']}")
            if data["sources"]:
                print(f"  Sources: {', '.join(set(data['sources']))}")
            # Display metrics
            latency = data.get("latency_seconds", 0.0)
            cost = data.get("total_cost_usd", 0.0)
            router_in = data.get("router_input_tokens", 0)
            router_out = data.get("router_output_tokens", 0)
            reason_in = data.get("reasoning_input_tokens", 0)
            reason_out = data.get("reasoning_output_tokens", 0)
            print(
                f"  Metrics: {latency:.2f}s | ${cost:.6f} | "
                f"Router tokens: {router_in} in / {router_out} out | "
                f"Reasoning tokens: {reason_in} in / {reason_out} out"
            )
            print()
        except Exception as e:
            print(f"\nError: {e}\n")

    # Consolidation step on exit
    print("\nConsolidating session memory, please wait...")
    try:
        resp = requests.post(CONSOLIDATE_URL, json={"session_id": session_id}, timeout=30)
        if resp.status_code == 200:
            res_data = resp.json()
            print("\n==================================================")
            print("             CONSOLIDATION SUMMARY")
            print("==================================================")
            print(f"Summary:       {res_data['summary']}")
            print(f"Facts Updated: {res_data['facts_updated']}")
            print("==================================================\n")
        else:
            print(f"Consolidation server response error: {resp.status_code}")
    except Exception as e:
        print(f"Consolidation failed: {e}")


if __name__ == "__main__":
    main()
