"""Cascade popularity prediction: MLP on early-spread features (baseline) vs a
CV-tuned LightGBM on richer early-adoption features (improved).
Cascades are simulated with an independent-cascade process on a Barabasi-Albert graph."""
import json, os, sys, time
import numpy as np
import networkx as nx
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

SEED = 42
SEEDS = [42, 43, 44]          # seeds for randomized models (baseline and improved)
N_NODES, N_CASCADES = 2000, 4000
WINDOW = 3                    # observe time steps t = 0..3 only
T0 = time.time()

def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)

def simulate_ic(adj, p, seeds, rng, max_steps=30):
    """Independent cascade: each newly active node gets one chance to activate
    each inactive neighbour with probability p. Returns activation time (-1 = inactive)."""
    t_act = np.full(len(adj), -1)
    t_act[seeds] = 0
    frontier, t = list(seeds), 0
    while frontier and t < max_steps:
        t += 1
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if t_act[v] < 0 and rng.random() < p:
                    t_act[v] = t
                    nxt.append(v)
        frontier = nxt
    return t_act

def features(t_act, deg, rich):
    """Features from the observation window only (no leakage of later adoption).
    Baseline: cumulative adopter counts at t=0..WINDOW.
    Improved: adds early-adopter degree stats, growth rates and time-to-k adopters."""
    early = np.where((t_act >= 0) & (t_act <= WINDOW))[0]
    cum = np.cumsum(np.bincount(t_act[early], minlength=WINDOW + 1)).astype(float)
    f = list(cum)
    if rich:
        d = deg[early]
        f += [d.mean(), d.max(), d.sum(), d.std()]          # early-adopter degree stats
        f += list(cum[1:] / cum[:-1])                        # growth rates per step
        times = np.sort(t_act[early])
        for k in (5, 10, 20):                                # time to k adopters (censored)
            f.append(times[k - 1] if len(times) >= k else WINDOW + 1)
    return np.array(f, dtype=float)

def build_dataset():
    """Simulate cascades on one BA graph; label = log(final size)."""
    rng = np.random.default_rng(SEED)
    G = nx.barabasi_albert_graph(N_NODES, 3, seed=SEED)
    adj = [list(G.neighbors(i)) for i in range(N_NODES)]
    deg = np.array([G.degree(i) for i in range(N_NODES)], dtype=float)
    Xb, Xr, y = [], [], []
    for _ in range(N_CASCADES):
        p = rng.uniform(0.01, 0.2)                           # per-cascade spreading probability
        seeds = rng.choice(N_NODES, size=rng.integers(1, 4), replace=False)
        t_act = simulate_ic(adj, p, seeds, rng)
        y.append(np.log((t_act >= 0).sum()))
        Xb.append(features(t_act, deg, False))
        Xr.append(features(t_act, deg, True))
    return np.array(Xb), np.array(Xr), np.array(y)

def evaluate(y_true, y_pred):
    return {"r2": r2_score(y_true, y_pred),
            "spearman": spearmanr(y_true, y_pred).correlation,
            "mae": mean_absolute_error(y_true, y_pred)}

FIXED_LGB = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

def main():
    os.makedirs("figures", exist_ok=True)
    log("Simulating independent cascades on a Barabasi-Albert graph...")
    Xb, Xr, y = build_dataset()
    if not (np.isfinite(y).all() and np.isfinite(Xr).all()):
        log("Non-finite data; cannot produce metrics."); sys.exit(1)
    # Single split, fixed seed; test set touched only at the end.
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED)
    log(f"Cascades: {len(y)} | train {len(idx_tr)} | test {len(idx_te)}")

    # Baseline: standard MLP on early-spread features (fixed hyper-parameters, no tuning).
    log("Baseline: MLPRegressor on early-spread features...")
    base = []
    for s in SEEDS:
        m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3,
                          max_iter=500, early_stopping=True, random_state=s))
        m.fit(Xb[idx_tr], y[idx_tr])
        base.append(evaluate(y[idx_te], m.predict(Xb[idx_te])))

    # Improved: LightGBM tuned by 3-fold CV on the training set only.
    log("Improved: CV-tuning LightGBM on training data only...")
    grid = {"num_leaves": [15, 31], "learning_rate": [0.03, 0.1],
            "n_estimators": [200, 400], "min_child_samples": [10, 30]}
    gs = GridSearchCV(lgb.LGBMRegressor(random_state=SEED, **FIXED_LGB), grid,
                      cv=3, scoring="r2", n_jobs=1).fit(Xr[idx_tr], y[idx_tr])
    best = gs.best_params_
    log(f"Best CV params: {best} (CV R2 {gs.best_score_:.4f})")
    imp = []
    for s in SEEDS:
        m = lgb.LGBMRegressor(random_state=s, **FIXED_LGB, **best).fit(Xr[idx_tr], y[idx_tr])
        imp.append(evaluate(y[idx_te], m.predict(Xr[idx_te])))

    def agg(runs, k):
        v = np.array([r[k] for r in runs])
        return float(v.mean()), float(v.std())
    b_r2, b_r2_sd = agg(base, "r2"); i_r2, i_r2_sd = agg(imp, "r2")
    b_sp, _ = agg(base, "spearman"); i_sp, _ = agg(imp, "spearman")
    b_mae, _ = agg(base, "mae"); i_mae, _ = agg(imp, "mae")
    if not np.isfinite([b_r2, i_r2]).all():
        log("Non-finite scores; exiting."); sys.exit(1)
    log(f"Test R2: baseline {b_r2:.4f} vs improved {i_r2:.4f}")

    results = {
        "question": "Can a cross-validation-tuned LightGBM model on richer early-adoption features "
                    "predict final cascade popularity better than a neural network on basic early-spread "
                    "features, when both are tested on the same held-out simulated cascades?",
        "method": "Independent-cascade simulation on a Barabasi-Albert graph (networkx); label = log final size; "
                  "single 80/20 split (seed 42); MLP baseline vs CV-tuned LightGBM; 3 seeds each.",
        "dataset": {"name": "Simulated independent-cascade cascades on a Barabasi-Albert graph (networkx)",
                    "source_url": "https://networkx.org/documentation/stable/reference/generated/"
                                  "networkx.generators.random_graphs.barabasi_albert_graph.html",
                    "n_rows": int(len(y)), "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "R2 on log(final cascade size) on held-out test set",
        "higher_is_better": True,
        "baseline": "scikit-learn MLPRegressor (64-32 hidden, standardized inputs) on early-spread features: "
                    "cumulative adopter counts over the first 4 time steps. Stand-in for the paper's neural encoder.",
        "improved": "LightGBM regressor tuned by 3-fold CV grid search on the training set, using the same early "
                    "counts plus early-adopter degree statistics, growth rates and time-to-k-adopters.",
        "metrics": {
            "baseline_score": b_r2, "improved_score": i_r2, "improvement": i_r2 - b_r2,
            "baseline_score_std": b_r2_sd, "improved_score_std": i_r2_sd,
            "baseline_spearman": b_sp, "improved_spearman": i_sp,
            "baseline_mae": b_mae, "improved_mae": i_mae,
            "lgbm_cv_r2": float(gs.best_score_),
        },
        "notes": f"Simulated data, not the paper's Twitter/Digg cascades; the paper's transformer+GCN model is "
                 f"not reproduced (MLP stand-in). Single graph and single split. Baseline uses fixed "
                 f"hyper-parameters, not tuned. Best LightGBM CV params: {best}. Spearman and MAE are "
                 f"secondary and not used for selection.",
    }
    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Baseline (MLP)", "Improved (LGBM)"], [b_r2, i_r2], yerr=[b_r2_sd, i_r2_sd], capsize=6,
           color=["grey", "steelblue"])
    ax.set_ylabel("Test R2 (log final size)"); ax.set_title("Mean over seeds 42-44 (error bars = std)")
    fig.tight_layout(); fig.savefig("figures/r2_comparison.png", dpi=120)
    log("Wrote results.json and figures/r2_comparison.png")

if __name__ == "__main__":
    main()
