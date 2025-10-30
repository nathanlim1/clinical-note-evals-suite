from typing import Any, Dict, List
from note_eval.preprocess import normalize_text, number_sentences_for_prompt


def build_judge_messages(claim_text: str, spans: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Construct messages for the judge LLM given a claim and evidence spans."""
    blocks: List[str] = []
    for sp in spans:
        t = normalize_text(sp["text"]) if sp.get("text") else ""
        numbered = sp.get("numbered") or number_sentences_for_prompt(t)
        blocks.append(f"{sp['span_id']}:\n{numbered}")

    user = (
        f'FULL CLAIM:\n"{claim_text}"\n\n'
        "EVIDENCE (with locally numbered sentences [1],[2],...):\n"
        + "\n\n".join(blocks)
    )

    return [
        {
            "role": "system",
            "content": (
                "Role: clinical fact checker for SOAP notes. Use ONLY provided transcript spans to verify a full claim (including all sub-claims); "
                "NO outside knowledge. Be strict—if a claim is medically incorrect/misleading or if evidence is weak/ambiguous, mark Unsupported. "
                "If a claim is only implied by the transcript and not explicit, also mark Unsupported. Assume proper names are correct. "
                "Decide one label: Supported, Contradicted, or Unsupported "
                "Respond with JSON only (no prose, no code fences).\n"
                "Schema: {label:'Supported'|'Contradicted'|'Unsupported', citations:[{span_id, sentences:[int]}], "
                "rationale:string(<=150), severity:'high'|'medium'|'low'|'none'}.\n"
                "Rules: If Supported/Contradicted, provide citations of all sentences that support/contradict the full claim as a list of "
                "{span_id, sentences:[ints]} using local numbers for that span; if Unsupported, citations must be []. "
                "Severity reflects clinical risk of incorrect/unsupported claims: high (diagnoses/medications/medication dosages/allergies/acute safety/plan changes); "
                "medium (material but non-urgent); low (minor context). Mark severity 'none' if the claim is Supported."
            ),
        },
        {"role": "user", "content": user},
    ]


def execute_judge_call(client: Any, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    def call():
        return client.chat_completion_json(messages, model)
    try:
        j = client.with_backoff(call, max_tries=3, base=1.5)
        label = j.get("label", "Unsupported")
        citations = j.get("citations", []) or []
        rationale = j.get("rationale", "")
        severity = j.get("severity", "none" if label == "Supported" else "medium")
    except Exception as e:
        label = "Unsupported"; citations = []
        rationale = f"judge_error: {e}"
        severity = "medium"
    return {"label": label, "citations": citations, "rationale": rationale, "severity": severity}
