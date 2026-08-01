"""
Evaluation Runner Script.
Runs the 50-query evaluation suite and reports intent classification accuracy,
reflex routing accuracy, routing decisions, retrieval precision/recall, latency, and cost.

Usage:
    python scripts/run_evals.py [--api] [--in-process] [--skip-reasoning]
"""
import os
import sys
import json
import time
import asyncio
import argparse
import requests
import numpy as np

# Ensure root dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import ROUTER_MODEL, REASONING_MODEL, MODEL_PRICING
from app.orchestrator.workflow import run_workflow

EVAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "../tests/eval_dataset.json")
API_URL = "http://127.0.0.1:8000/chat"


def run_query_in_process(query: str, skip_reasoning: bool = False):
    # Runs the actual production workflow directly in-process.
    # run_workflow is async (agent_node awaits execute_tool calls), so it's
    # driven with asyncio.run here to keep this a plain sync function for
    # the rest of the eval script.
    return asyncio.run(run_workflow(query, [], skip_reasoning=skip_reasoning))


def run_query_api(query: str):
    import uuid
    session_id = str(uuid.uuid4())
    resp = requests.post(API_URL, json={"session_id": session_id, "message": query})
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="Force API-based evaluation")
    parser.add_argument("--in-process", action="store_true", help="Force in-process evaluation")
    parser.add_argument("--skip-reasoning", action="store_true", help="Skip the reasoning model call to save cost/time")
    args = parser.parse_args()
    
    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"Error: dataset file not found at {EVAL_DATASET_PATH}")
        sys.exit(1)
        
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} evaluation queries.")
    
    # Determine mode
    mode = "in-process"
    if args.api:
        mode = "api"
    elif args.in_process:
        mode = "in-process"
    else:
        # Auto-detect if API is running
        try:
            resp = requests.get(API_URL.replace("/chat", "/health"), timeout=1.0)
            if resp.status_code == 200:
                mode = "api"
                print("Detected running server. Using API-based evaluation.")
        except requests.RequestException:
            print("No running server detected. Defaulting to in-process evaluation.")
            
    # Run evals
    results = []
    
    correct_rag_decisions = 0
    correct_route_decisions = 0
    correct_intent_decisions = 0
    correct_reflex_decisions = 0
    
    precision_scores = []
    recall_scores = []
    
    latencies = []
    costs = []
    
    print("\nRunning evaluations...")
    for idx, item in enumerate(dataset):
        query = item["query"]
        exp_needs_rag = item["expected_needs_rag"]
        exp_route = item["expected_route"]
        exp_intent = item["expected_intent"]
        exp_is_reflex = item["expected_is_reflex"]
        exp_sources = item["expected_sources"]
        
        print(f"[{idx+1}/{len(dataset)}] Query: {query[:50]}...")
        time.sleep(3.0)
        
        try:
            if mode == "api":
                res = run_query_api(query)
            else:
                res = run_query_in_process(query, skip_reasoning=args.skip_reasoning)
                
            actual_needs_rag = res.get("used_rag", False)
            actual_route = res.get("route", "direct_answer")
            actual_sources = res.get("sources", [])
            actual_intent = res.get("intent", "general_chat")
            actual_is_reflex = res.get("is_reflex", False)
            
            # Evaluate Classifications
            is_rag_correct = (actual_needs_rag == exp_needs_rag)
            is_route_correct = (actual_route == exp_route)
            is_intent_correct = (actual_intent == exp_intent)
            is_reflex_correct = (actual_is_reflex == exp_is_reflex)
            
            if is_rag_correct:
                correct_rag_decisions += 1
            if is_route_correct:
                correct_route_decisions += 1
            if is_intent_correct:
                correct_intent_decisions += 1
            if is_reflex_correct:
                correct_reflex_decisions += 1
                
            # Evaluate Retrieval
            if exp_needs_rag:
                exp_set = set(exp_sources)
                act_set = set(actual_sources)
                
                intersection = exp_set.intersection(act_set)
                
                # Precision
                if len(act_set) > 0:
                    precision = len(intersection) / len(act_set)
                else:
                    precision = 0.0
                precision_scores.append(precision)
                
                # Recall
                if len(exp_set) > 0:
                    recall = len(intersection) / len(exp_set)
                else:
                    recall = 1.0
                recall_scores.append(recall)
            else:
                if not actual_needs_rag:
                    pass
                else:
                    precision_scores.append(0.0)
                    recall_scores.append(0.0)
            
            latencies.append(res["latency_seconds"])
            costs.append(res["total_cost_usd"])
            
            results.append({
                "query": query,
                "expected": {
                    "needs_rag": exp_needs_rag,
                    "route": exp_route,
                    "intent": exp_intent,
                    "is_reflex": exp_is_reflex,
                    "sources": exp_sources
                },
                "actual": {
                    "needs_rag": actual_needs_rag,
                    "route": actual_route,
                    "intent": actual_intent,
                    "is_reflex": actual_is_reflex,
                    "sources": actual_sources,
                    "reply": res.get("reply", "")
                },
                "metrics": {
                    "latency": res["latency_seconds"],
                    "cost": res["total_cost_usd"],
                    "router_tokens": {
                        "in": res.get("router_input_tokens", 0),
                        "out": res.get("router_output_tokens", 0)
                    },
                    "reasoning_tokens": {
                        "in": res.get("reasoning_input_tokens", 0),
                        "out": res.get("reasoning_output_tokens", 0)
                    }
                }
            })
            
        except Exception as e:
            print(f"  FAILED: {e}")
            
    # Metrics summaries
    total_runs = len(results)
    if total_runs == 0:
        print("\nAll evaluation runs failed!")
        return
        
    rag_accuracy = correct_rag_decisions / total_runs
    route_accuracy = correct_route_decisions / total_runs
    intent_accuracy = correct_intent_decisions / total_runs
    reflex_accuracy = correct_reflex_decisions / total_runs
    
    avg_precision = np.mean(precision_scores) if precision_scores else 1.0
    avg_recall = np.mean(recall_scores) if recall_scores else 1.0
    
    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    total_cost = sum(costs)
    avg_cost = np.mean(costs)
    
    print("\n" + "="*50)
    print("               EVALUATION SUMMARY")
    print("="*50)
    print(f"Evaluation Mode:           {mode}")
    print(f"Queries Run:               {total_runs} / {len(dataset)}")
    print(f"Intent Classification Acc: {intent_accuracy*100:.1f}% ({correct_intent_decisions}/{total_runs})")
    print(f"Reflex Triage Accuracy:    {reflex_accuracy*100:.1f}% ({correct_reflex_decisions}/{total_runs})")
    print(f"Routing Accuracy (RAG?):   {rag_accuracy*100:.1f}% ({correct_rag_decisions}/{total_runs})")
    print(f"Routing Accuracy (Route):  {route_accuracy*100:.1f}% ({correct_route_decisions}/{total_runs})")
    print(f"Retrieval Precision:       {avg_precision*100:.1f}%")
    print(f"Retrieval Recall:          {avg_recall*100:.1f}%")
    print(f"Avg Latency:               {avg_latency:.2f}s")
    print(f"Median Latency:            {median_latency:.2f}s")
    print(f"95th Percentile Latency:   {p95_latency:.2f}s")
    print(f"Total Session Cost (USD):  ${total_cost:.6f}")
    print(f"Avg Cost per Query (USD):  ${avg_cost:.6f}")
    print("="*50)
    
    # Save report
    report = {
        "summary": {
            "mode": mode,
            "queries_run": total_runs,
            "intent_accuracy": intent_accuracy,
            "reflex_accuracy": reflex_accuracy,
            "routing_rag_accuracy": rag_accuracy,
            "routing_route_accuracy": route_accuracy,
            "retrieval_precision": avg_precision,
            "retrieval_recall": avg_recall,
            "latency": {
                "avg": avg_latency,
                "median": median_latency,
                "p95": p95_latency
            },
            "cost": {
                "total": total_cost,
                "avg": avg_cost
            }
        },
        "details": results
    }
    
    report_path = os.path.join(os.path.dirname(__file__), "../tests/eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved detailed evaluation report to {os.path.abspath(report_path)}")

if __name__ == "__main__":
    main()
