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


def pseudo_first_order_qt(t, qe, k1):
    t = np.asarray(t, dtype=float)
    qe = np.asarray(qe, dtype=float)
    k1 = np.asarray(k1, dtype=float)
    return qe * (1.0 - np.exp(-k1 * t))


def generate_pfo_dataset(experiment: dict, rng: np.random.Generator) -> pd.DataFrame:
    n_runs = experiment["N_RUNS"]
    times = np.array(experiment["times_min"], dtype=float)

    run_df = sample_run_inputs(experiment, rng, n_runs)

    C0 = run_df["C0_(mg/L)"].to_numpy()
    qe = run_df["qe_(mg/g)"].to_numpy()
    k1 = run_df["k1_(1/min)"].to_numpy()

    rows = []
    for i in range(n_runs):
        qt_true = pseudo_first_order_qt(times, qe=qe[i], k1=k1[i])

        qt_sd = experiment["noise"]["qt_sd_frac_of_qe"] * max(qe[i], 1e-6)
        qt_obs = qt_true + rng.normal(0.0, qt_sd, size=len(times))

        # keep uptake physically plausible
        qt_obs = np.clip(qt_obs, 0.0, qe[i])

        for tj, qtj in zip(times, qt_obs):
            rows.append({
                "run_id": i + 1,
                "t_(min)"   : float(tj),
                "C0_(mg/L)" : float(C0[i]),
                "qe_(mg/g)" : float(qe[i]),
                "qt_(mg/g)" : float(qtj),
                "k1_(1/min)": float(k1[i]),
            })

    return pd.DataFrame(rows).round(6)


def generate_timeseries_dataset(experiment: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Kept with the same function name so main.ipynb does not need to change.
    In this folder, it returns a pseudo-first-order kinetic dataset.
    """
    return generate_pfo_dataset(experiment, rng)


def stats_json(df: pd.DataFrame) -> str:
    df2 = df.copy()
    if "run_id" in df2.columns:
        df2 = df2.drop(columns=["run_id"])
    return json.dumps(
        df2.describe().T[["mean", "std", "min", "max"]].round(4).to_dict(),
        indent=2
    )