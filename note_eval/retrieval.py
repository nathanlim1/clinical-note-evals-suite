from typing import Any, Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray


def conversation_chunks(
    sent_list: List[Dict[str, Any]],
    min_chars: int = 160,
    max_chars: int = 800,
    min_sents: int = 2,
    max_sents: int = 8,
    response_min_sents: int = 1,
) -> List[Dict[str, Any]]:
    """
    Build conversation-aware chunks without relying on speaker labels.
    Heuristics:
      - Start a chunk at any sentence.
      - If a sentence ends with '?', include following sentences as the likely response
        until either another question appears (after at least response_min_sents), or limits hit.
      - Enforce min/max sentences and min/max characters.
      - Preserve global [start_char, end_char] from first to last sentence in the chunk.
    """
    chunks: List[Dict[str, Any]] = []
    num_sentences_total = len(sent_list)
    i = 0
    while i < num_sentences_total:
        start_idx = i
        end_idx = i
        num_sents = 1
        total_chars = len(sent_list[i]["text"]) if sent_list[i].get("text") else 0

        saw_question = sent_list[i]["text"].strip().endswith("?")
        responses_after_q = 0

        j = i + 1
        while j < num_sentences_total and num_sents < max_sents and total_chars < max_chars:
            sentence_text = sent_list[j]["text"].strip()
            is_question = sentence_text.endswith("?")

            if saw_question:
                if is_question and responses_after_q >= response_min_sents:
                    break
                if not is_question:
                    responses_after_q += 1

            next_len = total_chars + (len(sent_list[j]["text"]) if sent_list[j].get("text") else 0)
            if next_len > max_chars and num_sents >= min_sents:
                break

            end_idx = j
            num_sents += 1
            total_chars = next_len
            if is_question and not saw_question:
                saw_question = True
                responses_after_q = 0
            j += 1

        while end_idx + 1 < num_sentences_total and (num_sents < min_sents or total_chars < min_chars) and num_sents < max_sents:
            next_len = total_chars + (len(sent_list[end_idx+1]["text"]) if sent_list[end_idx+1].get("text") else 0)
            if next_len > max_chars:
                break
            end_idx += 1
            num_sents += 1
            total_chars = next_len

        text_slice = " ".join(s["text"] for s in sent_list[start_idx:end_idx+1])

        chunks.append({
            "chunk_id": f"C{start_idx}_{end_idx}",
            "text": text_slice,
            "start_idx": start_idx,
            "end_idx": end_idx,
        })

        i = end_idx + 1

    return chunks


def build_chunk_index(chunks: List[Dict[str, Any]], embedder: Any) -> Tuple[NDArray[np.float32], List[Dict[str, Any]]]:
    """Compute embeddings and a simple index for chunks.

    Returns:
        Tuple of (embeddings matrix, index_map list aligned to chunks).
    """
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)
    index_map: List[Dict[str, Any]] = [{
        "sent_idx": idx,
        "is_chunk": False,
        "chunk_start": 0,
        "chunk_end": len(chunks[idx]["text"]),
        "text": chunks[idx]["text"],
    } for idx in range(len(chunks))]
    return np.array(embeddings, dtype=np.float32), index_map


def retrieve_windows_for_claim(
    claim_text: str,
    sent_list: List[Dict[str, Any]],
    sent_emb: NDArray[np.float32],
    index_map: List[Dict[str, Any]],
    embedder: Any,
    k: int = 3,
) -> List[Dict[str, Any]]:
    """Return up to k evidence windows based on chunk similarity."""
    query_embedding = embedder.encode([claim_text])[0]
    sims = sent_emb @ query_embedding  # cosine because normalized

    top_indices = np.argsort(-sims)[:max(20, k*4)]

    sentence_scores = {}
    for idx in top_indices:
        entry = index_map[idx]
        sent_idx = entry["sent_idx"]
        score = float(sims[idx])
        if sent_idx not in sentence_scores or score > sentence_scores[sent_idx]:
            sentence_scores[sent_idx] = score

    top_chunk_indices = sorted(sentence_scores.keys(), key=lambda i: sentence_scores[i], reverse=True)[:k]

    windows: List[Dict[str, Any]] = []
    for sent_idx in top_chunk_indices:
        s_c = sent_list[sent_idx]
        text_slice = s_c.get("text", "")
        windows.append({
            "span_id": f"E{sent_idx}",
            "text": text_slice,
            "start_sent_idx": int(s_c.get("start_idx", sent_idx)),
            "end_sent_idx": int(s_c.get("end_idx", sent_idx)),
            "chunk_idx": int(sent_idx),
            "sim": float(sentence_scores[sent_idx])
        })
    return windows
