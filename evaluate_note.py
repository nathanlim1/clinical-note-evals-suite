import json, argparse
from note_eval.pipeline import evaluate


def main() -> None:
    """CLI entry point for running the evaluator from the command line."""
    ap = argparse.ArgumentParser(description="Evidence-bound LLM-as-Judge evaluator for SOAP note vs transcript")
    ap.add_argument("--transcript", required=True, help="Path to transcript .txt")
    ap.add_argument("--note", required=True, help="Path to SOAP note .txt")
    ap.add_argument("--out", default="individual_result.json", help="Output JSON path")

    # LLM / API
    ap.add_argument("--model", default="gpt-4.1-mini", help="Chat-completions model name (used for judge & criticality)")
    ap.add_argument("--api-key", default="", help="API key (or set env OPENAI_API_KEY)")

    # Retrieval / embedding
    ap.add_argument("--k", type=int, default=5, help="top-k transcript windows per claim")
    ap.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model")
    ap.add_argument("--retrieval-chunk-min-chars", type=int, default=160, help="Minimum characters per conversation chunk")
    ap.add_argument("--retrieval-chunk-max-chars", type=int, default=1000, help="Maximum characters per conversation chunk")
    ap.add_argument("--retrieval-chunk-min-sents", type=int, default=2, help="Minimum sentences per conversation chunk")
    ap.add_argument("--retrieval-chunk-max-sents", type=int, default=8, help="Maximum sentences per conversation chunk")
    ap.add_argument("--retrieval-response-min-sents", type=int, default=2, help="Min response sentences to include after a question")

    # Claim extraction rules
    ap.add_argument("--min-claim-chars", type=int, default=12, help="Minimum character length for a fragment to be considered a claim")

    # Judge concurrency / tokens
    ap.add_argument("--judge-concurrency", type=int, default=20, help="Max parallel LLM judge requests")

    # Criticality (LLM-based on uncovered only)
    ap.add_argument("--criticality-concurrency", type=int, default=20, help="Max parallel LLM criticality requests")

    # RAG Verification (for missing critical segments)
    ap.add_argument("--verification-k", type=int, default=5, help="Top-k note claims to retrieve for verifying each missing critical segment")
    ap.add_argument("--verification-concurrency", type=int, default=20, help="Max parallel LLM verification requests")

    args = ap.parse_args()

    with open(args.transcript, "r", encoding="utf-8") as f:
        transcript = f.read()
    with open(args.note, "r", encoding="utf-8") as f:
        note = f.read()

    out = evaluate(transcript, note, args)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
