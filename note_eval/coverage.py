from typing import Any, Dict, List, Set


def compute_covered_sentence_indices(
    results: List[Dict[str, Any]],
    topk_map: Dict[str, List[Dict[str, Any]]],
) -> Set[int]:
    """
    Map judge citations back to global transcript sentence indices.

    Args:
        results: List of per-claim results, including judge output
        topk_map: Mapping claim_id -> retrieval windows used for that claim

    Returns:
        Set of covered global sentence indices
    """
    covered_sent_indices: Set[int] = set()
    for r in results:
        claim_id = r.get("claim_id", "")
        if r.get("judge", {}).get("label") == "Supported":
            for cit in (r.get("judge", {}).get("citations") or []):
                span_id = cit.get("span_id")
                sp = next((s for s in (topk_map.get(claim_id) or []) if s.get("span_id") == span_id), None)
                if not sp:
                    continue
                a = int(sp.get("start_sent_idx", 0))
                b = int(sp.get("end_sent_idx", a))
                nums = cit.get("sentences", []) or []
                for n in nums:
                    try:
                        n_int = int(n)
                    except Exception:
                        continue
                    if n_int <= 0:
                        continue
                    g = a + (n_int - 1)
                    if g >= a and g <= b:
                        covered_sent_indices.add(g)
    return covered_sent_indices
