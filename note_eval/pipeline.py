import os, sys, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import dotenv
from note_eval.openai_client import OpenAIClient

from note_eval.preprocess import (
    note_to_claims,
    transcript_sentences,
    number_sentences_for_prompt,
)
from note_eval.retrieval import conversation_chunks, build_chunk_index, retrieve_windows_for_claim
from note_eval.judger import build_judge_messages, execute_judge_call
from note_eval.coverage import compute_covered_sentence_indices
from note_eval.criticality import build_criticality_messages, execute_criticality_call
from note_eval.verification import (
    retrieve_note_claims_for_segment,
    build_verification_messages,
    execute_verification_call,
)


dotenv.load_dotenv()


def evaluate(transcript_text: str, note_text: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Run the end-to-end evaluation pipeline and return results as a dict."""
    # 1) Prepare segments
    claims = note_to_claims(note_text, min_claim_chars=args.min_claim_chars)

    # 2) Require OpenAI API key and use OpenAI embeddings via OpenAIClient
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "") or os.getenv("LLM_API_KEY", "")
    if not api_key:
        print("ERROR: No API key provided. Set OPENAI_API_KEY/LLM_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(2)
    client = OpenAIClient(api_key=api_key, embedding_model=getattr(args, 'embedding_model', 'text-embedding-3-small'))
    embedder = client

    # 3) Build conversation-aware chunks for retrieval; keep sentence list for coverage
    t_sents = transcript_sentences(transcript_text)
    chunk_min_chars = getattr(args, 'retrieval_chunk_min_chars', 160)
    chunk_max_chars = getattr(args, 'retrieval_chunk_max_chars', 800)
    chunk_min_sents = getattr(args, 'retrieval_chunk_min_sents', 2)
    chunk_max_sents = getattr(args, 'retrieval_chunk_max_sents', 8)
    response_min_sents = getattr(args, 'retrieval_response_min_sents', 1)
    t_chunks = conversation_chunks(
        t_sents,
        min_chars=chunk_min_chars,
        max_chars=chunk_max_chars,
        min_sents=chunk_min_sents,
        max_sents=chunk_max_sents,
        response_min_sents=response_min_sents,
    )
    for c in t_chunks:
        c["numbered"] = number_sentences_for_prompt(c.get("text", "")) if c.get("text") else ""
    chunk_emb, index_map = build_chunk_index(t_chunks, embedder)

    # 4) Retrieve compact windows per claim (chunk-level)
    topk_map: Dict[str, List[Dict[str, Any]]] = {}
    numbered_cache: Dict[str, str] = {}
    for c in claims:
        wins = retrieve_windows_for_claim(
            c["text"], t_chunks, chunk_emb, index_map, embedder,
            k=args.k
        )
        topk_map[c["claim_id"]] = wins

    # 5) Judge + gating (parallelize LLM calls where applicable)
    results_map: Dict[str, Dict[str, Any]] = {}
    pending: List[Dict[str, Any]] = []

    for c in claims:
        claim_id = c["claim_id"]
        claim_text = c["text"]
        topk = topk_map[claim_id]

        spans_for_prompt = []
        for sp in topk:
            chunk_idx = sp.get("chunk_idx")
            if chunk_idx is not None and 0 <= chunk_idx < len(t_chunks):
                num = t_chunks[chunk_idx].get("numbered", "")
            else:
                key = f"{sp['span_id']}|{len(sp['text'])}"
                num = numbered_cache.get(key)
                if not num:
                    num = number_sentences_for_prompt(sp["text"])
                    numbered_cache[key] = num
            spans_for_prompt.append({"span_id": sp["span_id"], "text": sp["text"], "numbered": num})
        messages = build_judge_messages(claim_text, spans_for_prompt)
        pending.append({
            "claim_id": claim_id,
            "messages": messages
        })

        results_map[claim_id] = {
            "claim": c,
            "topk": topk,
            "judge": {"label": "Unsupported", "citations": [], "rationale": "", "severity": "none"}
        }

    if pending:
        with ThreadPoolExecutor(max_workers=args.judge_concurrency) as ex:
            futures = [ex.submit(lambda it=it: (it["claim_id"], execute_judge_call(client, it["messages"], args.model))) for it in pending]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="LLM judging",
                disable=not getattr(args, 'progress', True)
            ):
                claim_id, judge_out = fut.result()
                results_map[claim_id]["judge"] = judge_out

    results: List[Dict[str, Any]] = []
    for c in claims:
        claim_id = c["claim_id"]
        entry = results_map[claim_id]
        topk = entry["topk"]
        judge = entry["judge"]
        results.append({
            "claim_id": claim_id,
            "text": c["text"],
            "section": c["section"],
            "type": c["type"],
            "retrieval": [{"span_id": s["span_id"], "sim": s["sim"], "text": s["text"]} for s in topk],
            "judge": judge
        })

    # 6) Coverage mapping
    covered_sent_indices = compute_covered_sentence_indices(results, topk_map)

    def covered_by_index(idx: int) -> bool:
        return idx in covered_sent_indices

    # 7) Identify uncovered sentences; LLM classify only those
    uncovered = []
    for idx, ts in enumerate(t_sents):
        if not covered_by_index(idx):
            prev_text = t_sents[idx-1]["text"] if idx - 1 >= 0 else ""
            next_text = t_sents[idx+1]["text"] if idx + 1 < len(t_sents) else ""
            item = dict(ts)
            item["sent_idx"] = idx
            item["prev_text"] = prev_text
            item["next_text"] = next_text
            uncovered.append(item)

    crit_results: Dict[str, Dict[str, bool]] = {}
    missing_critical: List[Dict[str, Any]] = []

    if uncovered:
        with ThreadPoolExecutor(max_workers=args.criticality_concurrency) as ex:
            def submit_crit(ts):
                messages = build_criticality_messages(ts["text"], ts.get("prev_text", ""), ts.get("next_text", ""))
                crit = execute_criticality_call(client, messages, args.model)
                return ts["sent_id"], {"critical": crit}
            futures = [ex.submit(submit_crit, ts) for ts in uncovered]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="LLM criticality (missing only)",
                disable=not getattr(args, 'progress', True)
            ):
                sid, outc = fut.result()
                crit_results[sid] = outc

        missing_critical = [ts for ts in uncovered if crit_results.get(ts["sent_id"],{}).get("critical", False)]

    # 7b) RAG verification for missing critical segments against note claims
    verification_results: Dict[str, Dict[str, Any]] = {}
    verified_missing_critical: List[Dict[str, Any]] = []

    if missing_critical:
        with ThreadPoolExecutor(max_workers=args.verification_concurrency) as ex:
            def submit_ver(ts):
                retrieved_claims = retrieve_note_claims_for_segment(ts["text"], claims, embedder, k=args.verification_k)
                messages = build_verification_messages(ts["text"], retrieved_claims, ts.get("prev_text", ""), ts.get("next_text", ""))
                present, severity = execute_verification_call(client, messages, args.model)
                return ts["sent_id"], {"present": present, "severity": severity, "retrieved_claims": retrieved_claims}
            futures = [ex.submit(submit_ver, ts) for ts in missing_critical]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="RAG verification (missing critical)",
                disable=not getattr(args, 'progress', True)
            ):
                sid, outc = fut.result()
                verification_results[sid] = outc

        for ts in missing_critical:
            v = verification_results.get(ts["sent_id"], {})
            if not v.get("present", False):
                item = dict(ts)
                item["severity"] = v.get("severity", "medium")
                verified_missing_critical.append(item)
    else:
        verified_missing_critical = missing_critical

    # 8) Aggregates
    n = len(results) or 1
    unsupported_by_sev = {"low": 0, "medium": 0, "high": 0}
    contradicted_by_sev = {"low": 0, "medium": 0, "high": 0}
    for r in results:
        label = (r.get("judge", {}).get("label") or "").strip()
        sev = (r.get("judge", {}).get("severity") or "none").strip().lower()
        if sev not in unsupported_by_sev:
            continue
        if label == "Unsupported":
            unsupported_by_sev[sev] += 1
        elif label == "Contradicted":
            contradicted_by_sev[sev] += 1

    halluc_low = unsupported_by_sev["low"] + contradicted_by_sev["low"]
    halluc_med = unsupported_by_sev["medium"] + contradicted_by_sev["medium"]
    halluc_high = unsupported_by_sev["high"] + contradicted_by_sev["high"]

    token_usage = {}
    if client:
        token_usage = client.get_token_usage()

    # Compute missing critical severity breakdown
    mc_counts = {"low": 0, "medium": 0, "high": 0}
    for mc in verified_missing_critical:
        sev = (mc.get("severity") or "medium").lower()
        if sev in mc_counts:
            mc_counts[sev] += 1

    out: Dict[str, Any] = {
        "summary": {
            "claims": n,
            "hallucination_low_severity": halluc_low,
            "hallucination_medium_severity": halluc_med,
            "hallucination_high_severity": halluc_high,
            "unsupported_claim_low_severity": unsupported_by_sev["low"],
            "unsupported_claim_medium_severity": unsupported_by_sev["medium"],
            "unsupported_claim_high_severity": unsupported_by_sev["high"],
            "contradicted_claim_low_severity": contradicted_by_sev["low"],
            "contradicted_claim_medium_severity": contradicted_by_sev["medium"],
            "contradicted_claim_high_severity": contradicted_by_sev["high"],
            "missing_critical_count": len(verified_missing_critical),
            "missing_critical_low_severity": mc_counts["low"],
            "missing_critical_medium_severity": mc_counts["medium"],
            "missing_critical_high_severity": mc_counts["high"],
        },
        "token_usage": token_usage,
        "hallucinated": [r for r in results if r["judge"]["label"]=="Unsupported"],
        "contradicted": [r for r in results if r["judge"]["label"]=="Contradicted"],
        "missing_critical": verified_missing_critical,
        "claims": results,
        "transcript_text": transcript_text,
        "note_text": note_text,
    }
    return out
