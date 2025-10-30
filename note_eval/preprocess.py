import re
from typing import Any, Dict, List
import pysbd


SPACE_RE = re.compile(r"\s+")

SEC_HINTS = {
    "subjective": "S", "objective": "O", "assessment": "A", "plan": "P",
    "s:": "S", "o:": "O", "a:": "A", "p:": "P"
}

ENUM_ONLY_RE = re.compile(r'^\s*(?:\d+[\.\)]|[-*•])\s*$')


def normalize_text(text: str) -> str:
    """Normalize whitespace and non-breaking spaces in a string.

    Args:
        text: Input text which may contain irregular whitespace.

    Returns:
        A single-spaced, trimmed string with non-breaking spaces replaced.
    """
    return SPACE_RE.sub(" ", (text or "").replace("\u00A0", " ")).strip()


def sentences(text: str) -> List[str]:
    """Split text into sentences using pysbd (Pragmatic Segmenter)."""
    normalized_text = normalize_text(text)
    seg = pysbd.Segmenter(language="en", clean=False)
    return [sent.strip() for sent in seg.segment(normalized_text) if sent and sent.strip()]


def number_sentences_for_prompt(text: str) -> str:
    """Convert text into one sentence per line, each prefixed by [n]."""
    sents = sentences(text)
    return "\n".join(f"[{i+1}] {s.strip()}" for i, s in enumerate(sents))


def section_of(line: str, current: str = "UNK") -> str:
    """Infer section code from a line, falling back to the current section."""
    lowered_line = (line or "").strip().lower()
    for key, value in SEC_HINTS.items():
        if lowered_line.startswith(key):
            return value
    return current


def note_to_claims(note_text: str, min_claim_chars: int = 12) -> List[Dict[str, Any]]:
    """
    Build claims from the note.
    - Drops enumeration-only fragments (e.g., "1.", "2)")
    - Drops fragments shorter than min_claim_chars
    """
    claims: List[Dict[str, Any]] = []
    current_section = "UNK"
    claim_index = 0
    for raw_line in (note_text or "").splitlines():
        normalized_line = normalize_text(raw_line)
        if not normalized_line:
            continue
        new_section = section_of(normalized_line, current_section)
        if new_section != current_section and new_section != "UNK":
            current_section = new_section
            # drop the header token itself
            header = re.compile(r"^(subjective|objective|assessment|plan|[SOAP]:?)\s*[:\-]?\s*", re.I)
            normalized_line = header.sub("", normalized_line)
            if not normalized_line:
                continue
        for sent in sentences(normalized_line):
            atomic = normalize_text(sent)
            if not atomic:
                continue
            if ENUM_ONLY_RE.match(atomic):
                continue
            if len(atomic) < min_claim_chars:
                continue

            sections = {
                "S": "subjective", "O": "objective", "A": "assessment", "P": "plan"
            }

            claims.append({
                "claim_id": f"C{claim_index}",
                "text": atomic,
                "section": current_section,
                "type": sections[current_section]
            })
            claim_index += 1
    return claims


def transcript_sentences(text: str) -> List[Dict[str, Any]]:
    """Index sentences in the transcript without character offsets."""
    indexed_sentences: List[Dict[str, Any]] = []
    for sentence_text in sentences(text or ""):
        indexed_sentences.append({
            "sent_id": f"T{len(indexed_sentences)}",
            "text": sentence_text,
        })
    return indexed_sentences
