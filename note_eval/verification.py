from typing import Any, Dict, Iterable, List, Tuple
from note_eval.preprocess import normalize_text


def retrieve_note_claims_for_segment(segment_text: str, claims: List[Dict[str, Any]], embedder: Any, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most similar note claims for a given critical transcript segment.
    Returns list of claims sorted by similarity.
    """
    if not claims:
        return []

    segment_emb = embedder.encode([segment_text])[0]
    claim_texts = [c["text"] for c in claims]
    claim_embs = embedder.encode(claim_texts)

    sims = claim_embs @ segment_emb
    top_indices = sims.argsort()[::-1][:k]

    results = []
    for idx in top_indices:
        results.append({
            "claim_id": claims[idx]["claim_id"],
            "text": claims[idx]["text"],
            "section": claims[idx]["section"],
            "similarity": float(sims[idx])
        })
    return results


def build_verification_messages(critical_segment: str, note_claims: List[Dict[str, Any]], prev_sentence: str = "", next_sentence: str = "") -> List[Dict[str, str]]:
    """
    Verify if all key facts from TARGET are documented across the note claims (union of claims).
    """
    claims_text = "\n".join(f"{i+1}. \"{normalize_text(c['text'])}\"" for i, c in enumerate(note_claims))

    system = (
        "Role: verifier. Determine whether ALL key facts in TARGET are documented across the note claims (collectively). "
        "Use PREV/NEXT only for reference disambiguation. Respond with JSON only (no prose, no code fences). "
        "Schema: {present:true|false, severity:'high'|'medium'|'low'|'none'}. "
        "Criteria: present=true if every required fact in TARGET appears somewhere in one or more claims without contradiction. "
        "present=false if any required fact is missing, contradicted or uncertain/hedged. "
        "If present=true, set severity='none'. If present=false, set severity by clinical risk: high (diagnoses/medications/allergies/acute safety/plan changes); medium (material but not urgent); low (minor context)."
    )

    user = (
        "Semantic match allowed; wording may differ. Case-insensitive; ignore punctuation. Do NOT treat PREV/NEXT as content.\n\n"
        f"PREV: \"{(prev_sentence or '[none]')}\"\n"
        f"TARGET: \"{normalize_text(critical_segment)}\"\n"
        f"NEXT: \"{(next_sentence or '[none]')}\"\n\n"
        "CLAIMS:\n" + claims_text + "\n\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def execute_verification_call(client: Any, messages: List[Dict[str, str]], model: str) -> Tuple[bool, str]:
    def call():
        return client.chat_completion_json(messages, model)
    try:
        j = client.with_backoff(call, max_tries=3, base=1.5)
        present = bool(j.get("present", False))
        severity = j.get("severity", "none" if present else "medium")
        return present, severity
    except Exception:
        return False, "medium"
