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

def solve_Ce_from_langmuir_mass_balance(C0, qmax, KL, m, V):
    C0 = np.asarray(C0, dtype=float)
    qmax = np.asarray(qmax, dtype=float)
    KL = np.asarray(KL, dtype=float)
    m = np.asarray(m, dtype=float)
    V = np.asarray(V, dtype=float)

    a = KL
    b = 1.0 + (m / V) * qmax * KL - C0 * KL
    c = -C0

    disc = np.maximum(b*b - 4*a*c, 0.0)
    sqrt_disc = np.sqrt(disc)

    Ce1 = (-b + sqrt_disc) / (2*a)
    Ce2 = (-b - sqrt_disc) / (2*a)

    candidates = np.stack([Ce1, Ce2], axis=0)
    candidates = np.where(np.isfinite(candidates), candidates, np.nan)

    in_range = (candidates >= 0.0) & (candidates <= np.maximum(C0, 0.0))
    Ce = np.where(np.any(in_range, axis=0),
                  np.nanmax(np.where(in_range, candidates, np.nan), axis=0),
                  np.nan)

    nonneg = (candidates >= 0.0)
    Ce = np.where(np.isnan(Ce) & np.any(nonneg, axis=0),
                  np.nanmax(np.where(nonneg, candidates, np.nan), axis=0),
                  Ce)

    Ce = np.where(np.isnan(Ce), np.maximum(C0, 0.0), Ce)
    return Ce

def langmuir_qe(Ce, qmax, KL):
    Ce = np.asarray(Ce, dtype=float)
    return (qmax * KL * Ce) / (1.0 + KL * Ce)

def separation_factor_RL(C0, KL):
    C0 = np.asarray(C0, dtype=float)
    return 1.0 / (1.0 + KL * C0)

def pseudo_second_order_qt(t, qe, k2):
    t = np.asarray(t, dtype=float)
    qe = np.asarray(qe, dtype=float)
    k2 = np.asarray(k2, dtype=float)
    return (k2 * (qe**2) * t) / (1.0 + k2 * qe * t)

def generate_timeseries_dataset(experiment: dict, rng: np.random.Generator) -> pd.DataFrame:
    n_runs = experiment["N_RUNS"]
    times = np.array(experiment["times_min"], dtype=float)

    run_df = sample_run_inputs(experiment, rng, n_runs)

    C0   = run_df["C0_(mg/L)"].to_numpy()
    m    = run_df["m_(g)"].to_numpy()
    V    = run_df["V_(L)"].to_numpy()
    qmax = run_df["qmax_(mg/g)"].to_numpy()
    KL   = run_df["KL_(L/mg)"].to_numpy()
    k2   = run_df["k2_(1/(mg·min))"].to_numpy()

    Ce_true = solve_Ce_from_langmuir_mass_balance(C0=C0, qmax=qmax, KL=KL, m=m, V=V)
    qe_true = langmuir_qe(Ce_true, qmax=qmax, KL=KL)

    Ce_sd = experiment["noise"]["Ce_sd_frac_of_C0"] * np.maximum(C0, 1e-6)
    Ce_obs = Ce_true + rng.normal(0.0, Ce_sd, size=n_runs)
    Ce_obs = np.clip(Ce_obs, 0.0, C0)

    qe_obs = ((C0 - Ce_obs) * V) / m
    qe_obs = np.clip(qe_obs, 0.0, None)

    RL = separation_factor_RL(C0, KL)

    rows = []
    for i in range(n_runs):
        qt_true = pseudo_second_order_qt(times, qe=qe_true[i], k2=k2[i])

        qt_sd = experiment["noise"]["qt_sd_frac_of_qe"] * max(qe_true[i], 1e-6)
        qt_obs = qt_true + rng.normal(0.0, qt_sd, size=len(times))

        cap = max(qe_obs[i], 1e-9)
        qt_obs = np.clip(qt_obs, 0.0, cap)

        for tj, qtj in zip(times, qt_obs):
            rows.append({
                "run_id": i + 1,
                "t_(min)": float(tj),

                "C0_(mg/L)": float(C0[i]),
                "Ce_(mg/L)": float(Ce_obs[i]),
                "qe_(mg/g)": float(qe_obs[i]),
                "qt_(mg/g)": float(qtj),

                "qmax_(mg/g)": float(qmax[i]),
                "KL_(L/mg)"  : float(KL[i]),
                "k2_(1/(mg·min))": float(k2[i]),
                "RL_(-)"     : float(RL[i]),
                "m_(g)"      : float(m[i]),
                "V_(L)"      : float(V[i]),
            })

    return pd.DataFrame(rows).round(6)

def stats_json(df: pd.DataFrame) -> str:
    df2 = df.copy()
    if "run_id" in df2.columns:
        df2 = df2.drop(columns=["run_id"])
    return json.dumps(
        df2.describe().T[["mean", "std", "min", "max"]].round(4).to_dict(),
        indent=2
    )