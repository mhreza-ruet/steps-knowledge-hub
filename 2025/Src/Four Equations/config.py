# config.py
from dotenv import load_dotenv

def get_experiment():
    # Load .env (OPENAI_API_KEY)
    load_dotenv()

    experiment = {
        "title_rules": "Return a single concise title (8–14 words). No LaTeX commands, no quotes, no math.",

        "context": (
            "Write an observations-style report of phosphate adsorption and uptake kinetics in saline soils "
            "using batch tests. Each run records an initial phosphate concentration, an equilibrium concentration, "
            "an equilibrium adsorption capacity, and time-series uptake values. Adsorption behavior is summarized "
            "using an equilibrium isotherm framework and a kinetic uptake framework, and adsorption favorability "
            "is indicated using a standard dimensionless factor. Describe what was done and what was recorded, "
            "not how data were generated."
        ),
        "motivation": (
            "Provide a clear lab-note summary: state the number of runs, the number of time points per run, recorded ranges "
            "and summary statistics, and describe how uptake changed with time across runs with different conditions."
        ),

        "NUM_DATASETS": 4,
        "N_RUNS": 3,

        "times_min": [0, 2, 5, 10, 20, 30, 45, 60, 90, 120],

        "inputs": {
            "C0_(mg/L)"        : ("mg/L", "uniform", [5.0, 80.0]),
            "m_(g)"            : ("g",    "uniform", [0.20, 1.00]),
            "V_(L)"            : ("L",    "uniform", [0.020, 0.100]),
            "qmax_(mg/g)"      : ("mg/g", "uniform", [0.50, 10.0]),
            "KL_(L/mg)"        : ("L/mg", "uniform", [0.01, 0.50]),
            "k2_(1/(mg·min))"  : ("1/(mg·min)", "uniform", [0.005, 0.20]),
        },

        "equations": {
            "Langmuir_isotherm": "qe = (qmax * KL * Ce) / (1 + KL * Ce)",
            "Mass_balance"     : "C0 = Ce + (m/V) * qe",
            "Capacity_calc"    : "qe = ((C0 - Ce) * V) / m",
            "PSO_kinetics"     : "qt(t) = (k2 * qe^2 * t) / (1 + k2 * qe * t)",
            "Separation_factor": "RL = 1 / (1 + KL * C0)",
        },

        "noise": {
            "Ce_sd_frac_of_C0": 0.02,
            "qt_sd_frac_of_qe": 0.03},

        # Which notation families are allowed to be sampled (randomly) per report
        "notation_families": ["env_standard", "chem_eng", "materials", "soil_geo"],

        # Optional: force a specific family for debugging (set to "" for random)
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
        f"Use the actual number of time points per run T={len(experiment['times_min'])}.\n"
        f"Use the total number of recorded rows N_total={experiment['N_RUNS'] * len(experiment['times_min'])}.\n"
    )

    return experiment