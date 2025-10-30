import os, sys, csv, json, argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from note_eval.pipeline import evaluate


def load_pairs(pairs_csv: str):
    rows = []
    with open(pairs_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"transcript", "note"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must include at least 'transcript' and 'note' columns")
        for i, row in enumerate(reader):
            rows.append({
                "id": row.get("id") or str(i),
                "transcript": row["transcript"],
                "note": row["note"],
                "out": row.get("out", ""),
            })
    return rows


def build_eval_args(ns: argparse.Namespace) -> argparse.Namespace:
    # Build a namespace with the exact fields expected by evaluate()
    return argparse.Namespace(
        model=ns.model,
        api_key=ns.api_key,
        k=ns.k,
        embedding_model=ns.embedding_model,
        retrieval_chunk_min_chars=ns.retrieval_chunk_min_chars,
        retrieval_chunk_max_chars=ns.retrieval_chunk_max_chars,
        retrieval_chunk_min_sents=ns.retrieval_chunk_min_sents,
        retrieval_chunk_max_sents=ns.retrieval_chunk_max_sents,
        retrieval_response_min_sents=ns.retrieval_response_min_sents,
        min_claim_chars=ns.min_claim_chars,
        judge_concurrency=ns.judge_concurrency,
        criticality_concurrency=ns.criticality_concurrency,
        verification_k=ns.verification_k,
        verification_concurrency=ns.verification_concurrency,
        # Disable inner progress bars during batch runs for clarity
        progress=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch run clinical note evaluations over a CSV of pairs")
    ap.add_argument("--pairs-csv", required=True, help="CSV with columns: transcript,note[,id,out]")
    ap.add_argument("--out-dir", required=True, help="Directory to write per-item JSON outputs")
    ap.add_argument(
        "--aggregate-json",
        default="results.json",
        help="Path to write overall aggregate JSON (defaults to results.json)",
    )
    ap.add_argument("--max-workers", type=int, default=6, help="Max parallel evaluations")
    
    # Pass-through of evaluator options
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--retrieval-chunk-min-chars", type=int, default=160)
    ap.add_argument("--retrieval-chunk-max-chars", type=int, default=1000)
    ap.add_argument("--retrieval-chunk-min-sents", type=int, default=2)
    ap.add_argument("--retrieval-chunk-max-sents", type=int, default=8)
    ap.add_argument("--retrieval-response-min-sents", type=int, default=2)
    ap.add_argument("--min-claim-chars", type=int, default=12)
    ap.add_argument("--judge-concurrency", type=int, default=20)
    ap.add_argument("--criticality-concurrency", type=int, default=20)
    ap.add_argument("--verification-k", type=int, default=5)
    ap.add_argument("--verification-concurrency", type=int, default=20)

    ns = ap.parse_args()
    os.makedirs(ns.out_dir, exist_ok=True)

    pairs = load_pairs(ns.pairs_csv)
    eval_args = build_eval_args(ns)

    def run_one(item):
        # Accept either file paths or inline text for transcript/note
        def read_content(maybe_path: str) -> str:
            # If the string points to an existing file, read it; otherwise treat as inline text
            try:
                if maybe_path and os.path.exists(maybe_path) and os.path.isfile(maybe_path):
                    with open(maybe_path, "r", encoding="utf-8") as fh:
                        return fh.read()
            except Exception:
                # Fall back to treating as inline text if any filesystem check fails
                pass
            return maybe_path or ""

        transcript = read_content(item["transcript"])    
        note = read_content(item["note"])    
        result = evaluate(transcript, note, eval_args)
        out_path = item["out"] or os.path.join(ns.out_dir, f"{item['id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return item["id"], out_path, result.get("summary", {}), result.get("token_usage", {})

    summaries = []
    token_usage_totals = {}
    with ThreadPoolExecutor(max_workers=ns.max_workers) as ex:
        futures = [ex.submit(run_one, it) for it in pairs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Batch eval"):
            item_id, out_path, summary, t_usage = fut.result()
            summaries.append((item_id, out_path, summary))
            # Aggregate token usage if provided
            if isinstance(t_usage, dict):
                for k, v in t_usage.items():
                    try:
                        token_usage_totals[k] = token_usage_totals.get(k, 0) + v
                    except Exception:
                        # ignore non-numeric token usage entries
                        pass

    agg_json_path = ns.aggregate_json

    def sum_field(key: str) -> int:
        total = 0
        for _, _, s in summaries:
            total += int(s.get(key, 0) or 0)
        return total

    num_items = len(summaries)
    total_claims = sum_field("claims")
    halluc_low = sum_field("hallucination_low_severity")
    halluc_med = sum_field("hallucination_medium_severity")
    halluc_high = sum_field("hallucination_high_severity")
    unsupported_low = sum_field("unsupported_claim_low_severity")
    unsupported_med = sum_field("unsupported_claim_medium_severity")
    unsupported_high = sum_field("unsupported_claim_high_severity")
    contradicted_low = sum_field("contradicted_claim_low_severity")
    contradicted_med = sum_field("contradicted_claim_medium_severity")
    contradicted_high = sum_field("contradicted_claim_high_severity")
    missing_critical_total = sum_field("missing_critical_count")
    mc_low = sum_field("missing_critical_low_severity")
    mc_med = sum_field("missing_critical_medium_severity")
    mc_high = sum_field("missing_critical_high_severity")

    halluc_total = halluc_low + halluc_med + halluc_high
    unsupported_total = unsupported_low + unsupported_med + unsupported_high
    contradicted_total = contradicted_low + contradicted_med + contradicted_high

    def rate(x: int) -> float:
        return (float(x) / float(total_claims)) if total_claims > 0 else 0.0

    overall = {
        "num_items": num_items,
        "totals": {
            "claims": total_claims,
            "hallucination_low_severity": halluc_low,
            "hallucination_medium_severity": halluc_med,
            "hallucination_high_severity": halluc_high,
            "hallucination_total": halluc_total,
            "unsupported_claim_low_severity": unsupported_low,
            "unsupported_claim_medium_severity": unsupported_med,
            "unsupported_claim_high_severity": unsupported_high,
            "unsupported_total": unsupported_total,
            "contradicted_claim_low_severity": contradicted_low,
            "contradicted_claim_medium_severity": contradicted_med,
            "contradicted_claim_high_severity": contradicted_high,
            "contradicted_total": contradicted_total,
            "missing_critical_count": missing_critical_total,
            "missing_critical_low_severity": mc_low,
            "missing_critical_medium_severity": mc_med,
            "missing_critical_high_severity": mc_high,
        },
        "rates_per_claim": {
            "hallucination_low_severity_rate": rate(halluc_low),
            "hallucination_medium_severity_rate": rate(halluc_med),
            "hallucination_high_severity_rate": rate(halluc_high),
            "hallucination_rate": rate(halluc_total),
            "unsupported_low_severity_rate": rate(unsupported_low),
            "unsupported_medium_severity_rate": rate(unsupported_med),
            "unsupported_high_severity_rate": rate(unsupported_high),
            "unsupported_rate": rate(unsupported_total),
            "contradicted_low_severity_rate": rate(contradicted_low),
            "contradicted_medium_severity_rate": rate(contradicted_med),
            "contradicted_high_severity_rate": rate(contradicted_high),
            "contradicted_rate": rate(contradicted_total),
        },
        "averages_per_item": {
            "claims_per_item": (float(total_claims) / float(num_items)) if num_items else 0.0,
            "missing_critical_per_item": (float(missing_critical_total) / float(num_items)) if num_items else 0.0,
            "hallucinations_per_item": (float(halluc_total) / float(num_items)) if num_items else 0.0,
            "unsupported_per_item": (float(unsupported_total) / float(num_items)) if num_items else 0.0,
            "contradicted_per_item": (float(contradicted_total) / float(num_items)) if num_items else 0.0,
        },
        "token_usage": token_usage_totals,
    }

    with open(agg_json_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
