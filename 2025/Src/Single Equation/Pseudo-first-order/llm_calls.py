# llm_calls.py
import re
from utils_json import json_from_llm


def llm_extended_body(client, experiment: dict, stats: str) -> str:
    include_full = experiment.get("INCLUDE_FULL_TABLE", False)

    table_rules = (
        "A summary table is provided as Table~\\ref{tab:summary}.\n"
        + (
            "A full recorded dataset table is included at the end as Table~\\ref{tab:full}.\n"
            "At least one sentence MUST explicitly state that the full recorded dataset is provided at the end "
            "(Table~\\ref{tab:full}).\n"
            if include_full
            else
            "No full dataset table is included; do NOT mention Table~\\ref{tab:full}.\n"
        )
        + "If any table is mentioned, it MUST be referenced only as Table~\\ref{<LABEL>}.\n"
        + "Never write 'Table 1' or 'Table 2' in plain text.\n"
    )

    mode_instructions = (
        "This dataset contains time-resolved adsorption uptake observations.\n"
        "Each run records an initial concentration, an equilibrium adsorption capacity, "
        "a kinetic rate constant, and uptake values measured at multiple time points.\n"
        "Discuss how uptake changed with time across runs.\n"
        "Do NOT describe equilibrium isotherm fitting, heterogeneous surface modeling, "
        "monolayer adsorption capacity, or concentration-only equilibrium behavior.\n"
        "Do NOT describe the study as a purely equilibrium adsorption test.\n"
        "Focus on early uptake, gradual approach toward equilibrium uptake, kinetic differences "
        "among runs, and time-dependent adsorption behavior.\n"
    )

    prompt = (
        f"{experiment['context']}\n\n"
        f"Motivation: {experiment['motivation']}\n\n"
        f"{experiment['constraints']}\n\n"
        "DATASET MODE RULES (STRICT):\n"
        f"{mode_instructions}\n"
        "TABLE RULES (STRICT):\n"
        f"{table_rules}\n"
        "Do NOT include equations or explicit functional forms.\n\n"
        f"Dataset summary statistics (JSON):\n{stats}\n\n"
        f"Write an ~{experiment['paper_words']}-word LaTeX write-up as connected paragraphs.\n"
        "Do NOT include section headings.\n"
        "Do NOT include any table environments.\n"
        "Use \\emph{} for variables when they are used (but only from paragraph 2 onward).\n"
        "Describe what was observed and recorded in a realistic experimental tone.\n"
        "Do not mention simulation, synthetic generation, scripting, or data generation procedures.\n"
    )

    res = client.chat.completions.create(
        model=experiment["model_id"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=experiment["tok_paper"],
        temperature=0.4,
    )
    return res.choices[0].message.content.strip()


def llm_table_caption(client, experiment: dict, table_no: int, df, table_kind: str) -> str:
    """
    Returns caption TEXT ONLY (no 'Table 1:' prefix).
    Style: caption fragment (no 'was/were/is/are').
    Also forces Table 1 ~ summary and Table 2 ~ full recorded dataset.
    """
    cols = list(df.columns)
    preview = df.head(5).to_dict(orient="records")

    if table_no == 1:
        required_phrase = "summary statistics"
        intent = "This is a SUMMARY table (mean, SD, min, max)."
    else:
        required_phrase = "full recorded dataset"
        intent = "This is the FULL recorded dataset table (all rows)."

    prompt = (
        "Return JSON ONLY with key 'caption'.\n"
        "Write ONE caption as a short FRAGMENT (8–16 words).\n"
        "IMPORTANT STYLE RULES:\n"
        "- Do NOT start with 'Table' and do NOT include any table number.\n"
        "- Do NOT use finite verbs like 'was', 'were', 'is', 'are', 'shows', 'presents', 'reported'.\n"
        "- Use a noun phrase / fragment style.\n"
        f"- The caption MUST include the phrase '{required_phrase}'.\n"
        "- Do NOT include variable lists or numeric ranges.\n"
        "- Do NOT say generated/simulated/synthetic.\n\n"
        f"Table number: {table_no}\n"
        f"Table intent: {intent}\n\n"
        f"Context:\n{experiment['context']}\n\n"
        f"Columns:\n{cols}\n\n"
        f"First 5 rows (preview):\n{preview}\n"
    )

    res = client.chat.completions.create(
        model=experiment["model_id"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.2,
    )

    data = json_from_llm(res.choices[0].message.content)
    cap = (data.get("caption") or "").strip()

    cap = re.sub(r"^\s*Table\s*\d+\s*[:.\-]\s*", "", cap, flags=re.IGNORECASE).strip()
    cap = re.sub(r"\b(was|were|is|are|shows|showing|presents|presenting|reported)\b", "", cap, flags=re.IGNORECASE)
    cap = re.sub(r"\s+", " ", cap).strip()

    if required_phrase.lower() not in cap.lower():
        cap = f"{cap}, {required_phrase}".strip(" ,")

    cap = cap.rstrip(".").strip()
    if not cap:
        cap = required_phrase

    return cap + "."


def llm_title(client, experiment: dict, body_text: str, stats: str, title_cue: str) -> str:
    prompt = (
        f"{experiment.get('title_rules','Return a concise title.')}\n\n"
        "Return ONE title only (8–14 words).\n"
        "The title must be technical, concise, and journal-style (ACS format).\n"
        "Do NOT use quotes, LaTeX, or equations.\n"
        "Do NOT start with 'Study of', 'Investigation of', or 'Analysis of'.\n"
        "Vary structure and wording across runs — avoid repeating common phrasing.\n"
        "Use different syntactic patterns (e.g., gerund phrase, noun phrase, mechanism-focused, or regime-focused).\n"
        "The title should reflect kinetic adsorption behavior, uptake progression with time, "
        "approach to equilibrium, or rate-dependent uptake trends.\n\n"
        f"Title cue (must influence wording naturally): {title_cue}\n\n"
        "The title should reflect the specific behavior implied by the dataset.\n\n"
        f"Context:\n{experiment['context']}\n\n"
        f"Dataset summary (JSON):\n{stats}\n\n"
        "Body draft (partial context):\n"
        + body_text[:2500]
    )

    res = client.chat.completions.create(
        model=experiment["model_id"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.75
    )
    title = res.choices[0].message.content.strip().replace("\n", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title if title else "Untitled"