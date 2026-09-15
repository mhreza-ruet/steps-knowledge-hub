# config.py
from dotenv import load_dotenv

def get_experiment():
    # Load .env (OPENAI_API_KEY)
    load_dotenv()

    experiment = {
        "active_equation": "Freundlich_isotherm",
        "paper_mode": "equilibrium_isotherm",

        "title_rules": "Return a single concise title (8–14 words). No LaTeX commands, no quotes, no math.",

        "context": (
            "Write an observations-style report of phosphate adsorption equilibrium in saline soils "
            "using batch adsorption tests. Each run records an initial phosphate concentration, an "
            "equilibrium concentration, and an equilibrium adsorption capacity. Adsorption behavior "
            "is interpreted using a heterogeneous equilibrium adsorption framework with an empirical "
            "capacity factor and an adsorption-intensity parameter. Describe what was done and what "
            "was recorded, not how data were generated."
        ),

        "motivation": (
            "Provide a clear lab-note summary: state the number of runs, the recorded ranges and summary "
            "statistics, and describe how equilibrium adsorption changed with concentration across runs. "
            "Focus on adsorption capacity, concentration dependence, and heterogeneous adsorption behavior."
        ),

        "NUM_DATASETS": 4,
        "N_RUNS": 10,

        # Kept only so main flow remains compatible
        "times_min": [0],

        "inputs": {
            "C0_(mg/L)"   : ("mg/L", "uniform", [5.0, 80.0]),
            "m_(g)"       : ("g",    "uniform", [0.20, 1.00]),
            "V_(L)"       : ("L",    "uniform", [0.020, 0.100]),
            "KF_(mg/g)"   : ("mg/g", "uniform", [0.05, 2.50]),
            "n_(-)"       : ("-",    "uniform", [1.2, 4.0]),
        },

        "equations": {
            "Freundlich_isotherm": "qe = KF * Ce^(1/n)",
            "Mass_balance": "C0 = Ce + (m/V) * qe",
            "Capacity_calc": "qe = ((C0 - Ce) * V) / m",
        },

        "noise": {
            "Ce_sd_frac_of_C0": 0.02,
        },

        "notation_families": ["env_standard", "chem_eng", "materials", "soil_geo"],
        "notation_family": "soil_geo",

        "INCLUDE_FULL_TABLE": True,

        "paper_words": 1600,
        "tok_paper": 3000,
        "model_id": "gpt-4.1",

        "out_dir": "Output",
        "rng_seed": None,
        "per_report_seed": True,
    }

    experiment["constraints"] = (
        "[INSTRUCTIONS—DO NOT INCLUDE THIS PARAGRAPH IN THE PAPER]\n"
        "Write in PASSIVE VOICE ONLY.\n"
        "Do NOT use any sectioning commands (no \\section, \\subsection, etc.).\n"
        "Do NOT use any LaTeX environments like abstract.\n"
        "The first paragraph must summarize the work WITHOUT using any variable names.\n"
        "From the second paragraph onward, variables may be used but must be italic and correctly subscripted.\n"
        "Do NOT include any equations or explicit functional forms.\n"
        "No bullet lists.\n"
        "If tables are referenced, they MUST use LaTeX refs only: Table~\\ref{<LABEL>}.\n"
        "Never write 'Table 1' or 'Table 2' in plain text.\n"
        "If a second table is present, it must be referenced as Table~\\ref{tab:full}.\n"
        "If a second table is not present, do not mention it.\n"
        f"Use the actual number of runs N={experiment['N_RUNS']}.\n"
        f"Use the actual number of recorded rows N_total={experiment['N_RUNS']}.\n"
    )

    return experiment