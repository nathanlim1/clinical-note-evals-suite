import os
import json
import argparse
from typing import Dict, Any, List, Tuple
import streamlit as st
import pandas as pd
import altair as alt


def read_summary_from_file(path: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {}) if isinstance(data, dict) else {}
        token_usage = data.get("token_usage", {}) if isinstance(data, dict) else {}
        file_id = os.path.splitext(os.path.basename(path))[0]
        return file_id, summary, token_usage
    except Exception:
        return os.path.basename(path), {}, {}


def load_summaries(evals_dir: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    token_totals: Dict[str, float] = {}
    for name in sorted(os.listdir(evals_dir)):
        if not name.lower().endswith(".json"):
            continue
        file_path = os.path.join(evals_dir, name)
        file_id, s, t_usage = read_summary_from_file(file_path)
        if not s:
            continue

        row = {
            "id": file_id,
            "path": file_path,
            "claims": int(s.get("claims", 0) or 0),
            "hallucination_low_severity": int(s.get("hallucination_low_severity", 0) or 0),
            "hallucination_medium_severity": int(s.get("hallucination_medium_severity", 0) or 0),
            "hallucination_high_severity": int(s.get("hallucination_high_severity", 0) or 0),
            "unsupported_claim_low_severity": int(s.get("unsupported_claim_low_severity", 0) or 0),
            "unsupported_claim_medium_severity": int(s.get("unsupported_claim_medium_severity", 0) or 0),
            "unsupported_claim_high_severity": int(s.get("unsupported_claim_high_severity", 0) or 0),
            "contradicted_claim_low_severity": int(s.get("contradicted_claim_low_severity", 0) or 0),
            "contradicted_claim_medium_severity": int(s.get("contradicted_claim_medium_severity", 0) or 0),
            "contradicted_claim_high_severity": int(s.get("contradicted_claim_high_severity", 0) or 0),
            "missing_critical_count": int(s.get("missing_critical_count", 0) or 0),
            "missing_critical_low_severity": int(s.get("missing_critical_low_severity", 0) or 0),
            "missing_critical_medium_severity": int(s.get("missing_critical_medium_severity", 0) or 0),
            "missing_critical_high_severity": int(s.get("missing_critical_high_severity", 0) or 0),
        }
        rows.append(row)
        if isinstance(t_usage, dict):
            for k, v in t_usage.items():
                try:
                    token_totals[k] = token_totals.get(k, 0) + float(v)
                except Exception:
                    pass
    df = pd.DataFrame(rows)
    return df, token_totals


def compute_totals(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "num_items": 0,
            "claims": 0,
            "halluc": {"low": 0, "med": 0, "high": 0, "total": 0},
            "unsupported": {"low": 0, "med": 0, "high": 0, "total": 0},
            "contradicted": {"low": 0, "med": 0, "high": 0, "total": 0},
            "missing_critical": 0,
            "missing_critical_by_sev": {"low": 0, "med": 0, "high": 0},
            "rates": {"halluc": 0.0, "unsupported": 0.0, "contradicted": 0.0},
            "averages": {
                "claims_per_item": 0.0,
                "missing_critical_per_item": 0.0,
                "hallucinations_per_item": 0.0,
                "unsupported_per_item": 0.0,
                "contradicted_per_item": 0.0,
            },
        }

    num_items = int(df.shape[0])
    total_claims = int(df["claims"].sum())

    h_low = int(df["hallucination_low_severity"].sum())
    h_med = int(df["hallucination_medium_severity"].sum())
    h_high = int(df["hallucination_high_severity"].sum())
    u_low = int(df["unsupported_claim_low_severity"].sum())
    u_med = int(df["unsupported_claim_medium_severity"].sum())
    u_high = int(df["unsupported_claim_high_severity"].sum())
    c_low = int(df["contradicted_claim_low_severity"].sum())
    c_med = int(df["contradicted_claim_medium_severity"].sum())
    c_high = int(df["contradicted_claim_high_severity"].sum())
    miss = int(df["missing_critical_count"].sum())
    mc_low = int(df["missing_critical_low_severity"].sum()) if "missing_critical_low_severity" in df.columns else 0
    mc_med = int(df["missing_critical_medium_severity"].sum()) if "missing_critical_medium_severity" in df.columns else 0
    mc_high = int(df["missing_critical_high_severity"].sum()) if "missing_critical_high_severity" in df.columns else 0

    h_tot = h_low + h_med + h_high
    u_tot = u_low + u_med + u_high
    c_tot = c_low + c_med + c_high

    rate = (lambda x: float(x) / float(total_claims)) if total_claims > 0 else (lambda x: 0.0)

    return {
        "num_items": num_items,
        "claims": total_claims,
        "halluc": {"low": h_low, "med": h_med, "high": h_high, "total": h_tot},
        "unsupported": {"low": u_low, "med": u_med, "high": u_high, "total": u_tot},
        "contradicted": {"low": c_low, "med": c_med, "high": c_high, "total": c_tot},
        "missing_critical": miss,
        "missing_critical_by_sev": {"low": mc_low, "med": mc_med, "high": mc_high},
        "rates": {"halluc": rate(h_tot), "unsupported": rate(u_tot), "contradicted": rate(c_tot)},
        "averages": {
            "claims_per_item": (float(total_claims) / float(num_items)) if num_items else 0.0,
            "missing_critical_per_item": (float(miss) / float(num_items)) if num_items else 0.0,
            "hallucinations_per_item": (float(h_tot) / float(num_items)) if num_items else 0.0,
            "unsupported_per_item": (float(u_tot) / float(num_items)) if num_items else 0.0,
            "contradicted_per_item": (float(c_tot) / float(num_items)) if num_items else 0.0,
        },
    }


def longform_counts(totals: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, key in [("Hallucinated", "halluc"), ("Unsupported", "unsupported"), ("Contradicted", "contradicted")]:
        part = totals.get(key, {})
        rows.append({"category": label, "severity": "Low", "count": int(part.get("low", 0))})
        rows.append({"category": label, "severity": "Medium", "count": int(part.get("med", 0))})
        rows.append({"category": label, "severity": "High", "count": int(part.get("high", 0))})
    return pd.DataFrame(rows)


def per_item_error_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["hallucinated_total"] = (
        out["hallucination_low_severity"] + out["hallucination_medium_severity"] + out["hallucination_high_severity"]
    )
    out["unsupported_total"] = (
        out["unsupported_claim_low_severity"]
        + out["unsupported_claim_medium_severity"]
        + out["unsupported_claim_high_severity"]
    )
    out["contradicted_total"] = (
        out["contradicted_claim_low_severity"]
        + out["contradicted_claim_medium_severity"]
        + out["contradicted_claim_high_severity"]
    )
    out["errors_total"] = out["hallucinated_total"] + out["unsupported_total"] + out["contradicted_total"]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evals-dir", required=True)
    # Streamlit will pass many of its own flags; tolerate unknowns
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()

    st.set_page_config(page_title="Clinical Note Evals", layout="wide")
    st.title("Generated SOAP Note Evaluations")
    st.caption(f"Nathan Lim - October 2025")
    st.divider()

    st.subheader("Overview")

    if not os.path.isdir(args.evals_dir):
        st.error("Provided evals directory not found")
        return

    df, token_totals = load_summaries(args.evals_dir)
    totals = compute_totals(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Note/Transcript Pairs Evaluated", totals["num_items"]) 
    c2.metric("Total claims", totals["claims"]) 
    c3.metric("Probability of a Claim Being Hallucinated", f"{totals['rates']['halluc']:.2%}")
    
    c4, c5, c6 = st.columns(3)
    c4.metric("Average Claims/Note", f"{totals['averages']['claims_per_item']:.2f}")
    c5.metric("Total Number of Missing Critical Facts", totals["missing_critical"]) 
    c6.metric("Average Hallucinated Claims/Note", f"{totals['averages']['hallucinations_per_item']:.2f}")

    st.divider()

    st.subheader("Hallucinated Claim Counts by Category and Severity")
    st.caption("High Severity: Diagnoses, medications, medication dosages, allergies, acute safety, or plan changes")
    st.caption("Medium Severity: Material but non-urgent clinical information")
    st.caption("Low Severity: Minor clinical context")
    counts_df = longform_counts(totals)
    if not counts_df.empty:
        chart_df = (
            counts_df
            .rename(columns={"count": "Count", "category": "Category", "severity": "Severity"})
        )
        chart_df = chart_df[chart_df["Category"].isin(["Unsupported", "Contradicted"])].copy()
        severity_order = ["Low", "Medium", "High"]
        category_order = ["Unsupported", "Contradicted"]
        chart_df["TypeSeverity"] = chart_df["Category"] + " - " + chart_df["Severity"]
        type_sev_order = [f"{cat} - {sev}" for cat in category_order for sev in severity_order]
        bars = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("TypeSeverity:N", sort=type_sev_order, title="Type - Severity", axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color(
                    "Category:N",
                    scale=alt.Scale(domain=category_order, range=["#4C78A8", "#F58518"]),
                    legend=alt.Legend(title="Type"),
                ),
                tooltip=[
                    alt.Tooltip("Category:N", title="Type"),
                    alt.Tooltip("Severity:N"),
                    alt.Tooltip("Count:Q"),
                ],
            )
        )
        st.altair_chart(bars, width='stretch')
    else:
        st.info("No errors found across items.")

    st.divider()

    st.subheader("Hallucinated Claim Type: Unsupported vs. Contradicted")
    st.caption("Unsupported Claim: A claim present in the generated note that is not supported by the transcript.")
    st.caption("Contradicted Claim: A claim in the generated note that directly conflicts with or is disproven by the transcript.")
    h_total = int(totals.get("halluc", {}).get("total", 0))
    u_total = int(totals.get("unsupported", {}).get("total", 0))
    c_total = int(totals.get("contradicted", {}).get("total", 0))
    unsupported_rate = (float(u_total) / float(h_total)) if h_total > 0 else 0.0
    contradicted_rate = (float(c_total) / float(h_total)) if h_total > 0 else 0.0
    pie_df = pd.DataFrame([
        {"Category": "Unsupported Claims", "Rate": unsupported_rate},
        {"Category": "Contradicted", "Rate": contradicted_rate},
    ])

    
    pie = (
        alt.Chart(pie_df)
        .mark_arc(innerRadius=0)
        .encode(
            theta=alt.Theta(field="Rate", type="quantitative"),
            color=alt.Color(field="Category", type="nominal"),
            tooltip=[
                alt.Tooltip("Category:N"),
                alt.Tooltip("Rate:Q", format=".2%"),
            ],
        )
    )

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.metric("Unsupported Claims / Hallucinated Claims:", f"{unsupported_rate:.2%}")
        st.metric("Contradicted Claims / Hallucinated Claims:", f"{contradicted_rate:.2%}")
    with right_col:
        st.altair_chart(pie, width='stretch')

    st.divider()
        
    st.subheader("Missing Critical Facts Severity")
    mc_tot = int(totals.get("missing_critical", 0))
    mc_by = totals.get("missing_critical_by_sev", {}) or {}
    mc_low = int(mc_by.get("low", 0))
    mc_med = int(mc_by.get("med", 0))
    mc_high = int(mc_by.get("high", 0))
    mc_low_rate = (float(mc_low) / float(mc_tot)) if mc_tot > 0 else 0.0
    mc_med_rate = (float(mc_med) / float(mc_tot)) if mc_tot > 0 else 0.0
    mc_high_rate = (float(mc_high) / float(mc_tot)) if mc_tot > 0 else 0.0

    mc_df = pd.DataFrame([
        {"Severity": "High", "Rate": mc_high_rate},
        {"Severity": "Medium", "Rate": mc_med_rate},
        {"Severity": "Low", "Rate": mc_low_rate},
    ])

    mc_pie = (
        alt.Chart(mc_df)
        .mark_arc(innerRadius=0)
        .encode(
            theta=alt.Theta(field="Rate", type="quantitative"),
            color=alt.Color(
                field="Severity",
                type="nominal",
                scale=alt.Scale(domain=["High", "Medium", "Low"], range=["#D62728", "#FF7F0E", "#FFD700"]),
                legend=alt.Legend(title="Severity"),
            ),
            tooltip=[
                alt.Tooltip("Severity:N"),
                alt.Tooltip("Rate:Q", format=".2%"),
            ],
        )
    )

    mc_left, mc_right = st.columns([1, 1])
    with mc_left:
        st.metric("Percent High Severity:", f"{mc_high_rate:.2%}")
        st.metric("Percent Medium Severity:", f"{mc_med_rate:.2%}")
        st.metric("Percent Low Severity:", f"{mc_low_rate:.2%}")
    with mc_right:
        st.altair_chart(mc_pie, width='stretch')

    st.divider()

    st.subheader("Distribution: Hallucinated Claims / Total Claims per Note")
    per_item = per_item_error_df(df)
    if not per_item.empty:
        per_item_nonzero = per_item[per_item["claims"] > 0].copy()
        if not per_item_nonzero.empty:
            per_item_nonzero["halluc_rate"] = per_item_nonzero["hallucinated_total"] / per_item_nonzero["claims"]
            per_item_nonzero["_label"] = ""
            box = (
                alt.Chart(per_item_nonzero)
                .mark_boxplot()
                .encode(
                    x=alt.X("halluc_rate:Q", title="hallucination rate", axis=alt.Axis(format=".0%")),
                    y=alt.Y("_label:N", title=""),
                    tooltip=[alt.Tooltip("halluc_rate:Q", title="Hallucination rate", format=".2%")],
                )
                .properties(height=80)
            )
            st.altair_chart(box, width='stretch')
        else:
            st.caption("No notes with non-zero claim counts to compute rates.")
    else:
        st.caption("No per-note data available.")

    st.divider()

    # Highest hallucination rate pair: show transcript and note
    if not per_item.empty:
        per_item_nonzero = per_item[per_item["claims"] > 0].copy()
        if not per_item_nonzero.empty:
            per_item_nonzero["halluc_rate"] = per_item_nonzero["hallucinated_total"] / per_item_nonzero["claims"]
            top_idx = per_item_nonzero["halluc_rate"].idxmax()
            if pd.notna(top_idx):
                top_row = per_item_nonzero.loc[top_idx]
                st.subheader("Highest Hallucination Rate Note/Transcript Pair")
                st.caption(f"ID: {top_row['id']} — percentage of claims hallucinated: {top_row['halluc_rate']:.2%} ({int(top_row['hallucinated_total'])}/{int(top_row['claims'])})")
                transcript_text = ""; note_text = ""
                try:
                    with open(str(top_row.get("path", "")), "r", encoding="utf-8") as f:
                        jd = json.load(f)
                    transcript_text = jd.get("transcript_text", transcript_text)
                    note_text = jd.get("note_text", note_text)
                except Exception:
                    pass
                if transcript_text or note_text:
                    cta, ctb = st.columns(2)
                    with cta:
                        st.text_area("Transcript", transcript_text or "(not available)", height=240)
                    with ctb:
                        st.text_area("Generated Note", note_text or "(not available)", height=240)
                else:
                    st.caption("Transcript and note not available in per-item output. Re-run generation to include them.")

                # Show detailed sections from JSON for this pair
                try:
                    hallucinated_list = jd.get("hallucinated", []) or []
                    contradicted_list = jd.get("contradicted", []) or []
                    missing_critical_list = jd.get("missing_critical", []) or []

                    with st.expander(f"Unsupported Claims ({len(hallucinated_list)})", expanded=False):
                        if hallucinated_list:
                            h_rows = []
                            for item in hallucinated_list:
                                judge = (item.get("judge") or {}) if isinstance(item, dict) else {}
                                h_rows.append({
                                    "claim_id": item.get("claim_id"),
                                    "text": item.get("text"),
                                    "section": item.get("section"),
                                    "severity": judge.get("severity"),
                                    "rationale": judge.get("rationale"),
                                })
                            if h_rows:
                                st.dataframe(pd.DataFrame(h_rows), hide_index=True, width='stretch')
                            else:
                                st.caption("No items to display.")
                        else:
                            st.caption("None")

                    with st.expander(f"Contradicted Claims ({len(contradicted_list)})", expanded=False):
                        if contradicted_list:
                            c_rows = []
                            for item in contradicted_list:
                                judge = (item.get("judge") or {}) if isinstance(item, dict) else {}
                                c_rows.append({
                                    "claim_id": item.get("claim_id"),
                                    "text": item.get("text"),
                                    "section": item.get("section"),
                                    "severity": judge.get("severity"),
                                    "rationale": judge.get("rationale"),
                                })
                            if c_rows:
                                st.dataframe(pd.DataFrame(c_rows), hide_index=True, width='stretch')
                            else:
                                st.caption("No items to display.")
                        else:
                            st.caption("None")

                    with st.expander(f"Missing Critical Facts ({len(missing_critical_list)})", expanded=False):
                        if missing_critical_list:
                            m_rows = []
                            for item in missing_critical_list:
                                m_rows.append({
                                    "sent_id": item.get("sent_id"),
                                    "text": item.get("text"),
                                    "severity": item.get("severity"),
                                })
                            if m_rows:
                                st.dataframe(pd.DataFrame(m_rows), hide_index=True, width='stretch')
                            else:
                                st.caption("No items to display.")
                        else:
                            st.caption("None")
                except Exception:
                    st.caption("Could not load hallucinated/contradicted/missing-critical details for this pair.")

    st.divider()

    st.subheader("OpenAI API usage (aggregated)")
    if token_totals:
        toks = pd.DataFrame(sorted(token_totals.items()), columns=["Metric", "Value"])
        st.dataframe(toks, hide_index=True, width='stretch')
    else:
        st.caption("No token usage data found in individual eval files.")


if __name__ == "__main__":
    main()
