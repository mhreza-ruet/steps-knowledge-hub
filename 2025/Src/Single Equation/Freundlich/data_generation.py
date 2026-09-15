# data_generation.py
import numpy as np
import pandas as pd
import json


def sample_run_inputs(experiment: dict, rng: np.random.Generator, n_runs: int) -> pd.DataFrame:
    dists = {
        "uniform": lambda a, b, n: rng.uniform(a, b, n),
        "normal":  lambda m, s, n: rng.normal(m, s, n),
    }
    return pd.DataFrame({
        col: dists[dist](*params, n_runs)
        for col, (_, dist, params) in experiment["inputs"].items()
    })


def freundlich_qe(Ce, KF, n):
    Ce = np.asarray(Ce, dtype=float)
    KF = np.asarray(KF, dtype=float)
    n = np.asarray(n, dtype=float)
    return KF * np.power(np.maximum(Ce, 0.0), 1.0 / n)


def solve_C0_from_freundlich_mass_balance(Ce, KF, n, m, V):
    Ce = np.asarray(Ce, dtype=float)
    KF = np.asarray(KF, dtype=float)
    n = np.asarray(n, dtype=float)
    m = np.asarray(m, dtype=float)
    V = np.asarray(V, dtype=float)

    qe = freundlich_qe(Ce, KF, n)
    C0 = Ce + (m / V) * qe
    return C0, qe


def generate_freundlich_dataset(experiment: dict, rng: np.random.Generator) -> pd.DataFrame:
    n_runs = experiment["N_RUNS"]
    run_df = sample_run_inputs(experiment, rng, n_runs)

    m  = run_df["m_(g)"].to_numpy()
    V  = run_df["V_(L)"].to_numpy()
    KF = run_df["KF_(mg/g)"].to_numpy()
    n  = run_df["n_(-)"].to_numpy()

    # We generate equilibrium concentration first, then compute qe and C0 from
    # Freundlich + mass balance so the dataset stays physically consistent.
    Ce_true = rng.uniform(0.5, 40.0, size=n_runs)

    C0_true, qe_true = solve_C0_from_freundlich_mass_balance(
        Ce=Ce_true, KF=KF, n=n, m=m, V=V
    )

    # Keep only physically plausible runs within configured C0 range
    C0_min, C0_max = experiment["inputs"]["C0_(mg/L)"][2]

    mask = (
        np.isfinite(C0_true) &
        np.isfinite(qe_true) &
        (Ce_true >= 0.0) &
        (qe_true >= 0.0) &
        (C0_true >= C0_min) &
        (C0_true <= C0_max) &
        (Ce_true <= C0_true)
    )

    # Resample until enough valid runs are collected
    rows = []
    while len(rows) < n_runs:
        needed = n_runs - len(rows)

        sub_df = sample_run_inputs(experiment, rng, needed)
        m_sub  = sub_df["m_(g)"].to_numpy()
        V_sub  = sub_df["V_(L)"].to_numpy()
        KF_sub = sub_df["KF_(mg/g)"].to_numpy()
        n_sub  = sub_df["n_(-)"].to_numpy()

        Ce_sub = rng.uniform(0.5, 40.0, size=needed)
        C0_sub, qe_sub = solve_C0_from_freundlich_mass_balance(
            Ce=Ce_sub, KF=KF_sub, n=n_sub, m=m_sub, V=V_sub
        )

        good = (
            np.isfinite(C0_sub) &
            np.isfinite(qe_sub) &
            (Ce_sub >= 0.0) &
            (qe_sub >= 0.0) &
            (C0_sub >= C0_min) &
            (C0_sub <= C0_max) &
            (Ce_sub <= C0_sub)
        )

        for i in range(needed):
            if not good[i]:
                continue

            # Add small observation noise to Ce, then recompute qe from capacity equation
            Ce_sd = experiment["noise"]["Ce_sd_frac_of_C0"] * max(C0_sub[i], 1e-6)
            Ce_obs = Ce_sub[i] + rng.normal(0.0, Ce_sd)
            Ce_obs = float(np.clip(Ce_obs, 0.0, C0_sub[i]))

            qe_obs = ((C0_sub[i] - Ce_obs) * V_sub[i]) / m_sub[i]
            qe_obs = float(max(qe_obs, 0.0))

            rows.append({
                "run_id": len(rows) + 1,
                "C0_(mg/L)" : float(C0_sub[i]),
                "Ce_(mg/L)" : Ce_obs,
                "qe_(mg/g)" : qe_obs,
                "KF_(mg/g)" : float(KF_sub[i]),
                "n_(-)"     : float(n_sub[i]),
                "m_(g)"     : float(m_sub[i]),
                "V_(L)"     : float(V_sub[i]),
            })

            if len(rows) >= n_runs:
                break

    return pd.DataFrame(rows).round(6)


def generate_timeseries_dataset(experiment: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Kept with the same function name so main.ipynb does not need to change.
    In this folder, it returns an equilibrium-only Freundlich dataset.
    """
    return generate_freundlich_dataset(experiment, rng)


def stats_json(df: pd.DataFrame) -> str:
    df2 = df.copy()
    if "run_id" in df2.columns:
        df2 = df2.drop(columns=["run_id"])
    return json.dumps(
        df2.describe().T[["mean", "std", "min", "max"]].round(4).to_dict(),
        indent=2
    )