from typing import Any, Dict, List, Tuple


def build_criticality_messages(sentence_text: str, prev_sentence: str = "", next_sentence: str = "") -> List[Dict[str, str]]:
    """Construct messages for the binary criticality classifier LLM."""
    system = (
        "Role: clinical criticality classifier. Decide if TARGET is clinically CRITICAL for THIS encounter's SOAP note.\n"
        "Use PREV/NEXT only for reference disambiguation. Respond with JSON only (no prose, no code fences).\n"
        "Schema: {critical:true|false}.\n"
        "Return true ONLY if TARGET explicitly states a new/changed medical FACT or DECISION, e.g.:\n"
        "- Diagnosis/problem identification or status change\n"
        "- Medications/allergies with action or change (start/stop/increase/decrease/refill), dose/route/frequency\n"
        "- Key symptoms WITH attributes (onset/duration/severity/worsening/improving) or new red flags\n"
        "- Exam/vitals/findings or quantified results (numbers/units/ranges)\n"
        "- Orders/plan/assessment/decision (labs, imaging, referrals, procedures, follow-up interval)\n"
        "Hard negatives (return false): questions/interrogatives; greetings/pleasantries/empathy; acknowledgments/backchannels; admin/logistics/scheduling; small talk; set-ups; repeats without new clinical info; fragments without medical content.\n"
        "If uncertain or the sentence lacks the above positive signals, return false."
    )
    user = (
        f"TARGET:{sentence_text}\nPREV (context only):{(prev_sentence or '[none]')}\nNEXT (context only):{(next_sentence or '[none]')}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",  "content": user},
    ]


def execute_criticality_call(client: Any, messages: List[Dict[str, str]], model: str) -> bool:
    def call():
        return client.chat_completion_json(messages, model)
    try:
        j = client.with_backoff(call, max_tries=3, base=1.5)
        return bool(j.get("critical", False))
    except Exception:
        return False
