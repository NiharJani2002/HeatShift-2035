"""
proofs.py - run `python proofs.py`.  Each test simulates data where the TRUE answer is known and checks that the
method recovers it.  Nothing here uses Kaggle data, so the proofs are independent of the datasets.
"""
import sys, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
import mathcore as mc


def P1_conformal(reps=25):
    cov_c, cov_n = [], []
    for r in range(reps):
        rng = np.random.default_rng(r)
        G, per = 24, 30
        g = np.repeat(np.arange(G), per)
        X = rng.random((G * per, 5)); gx = rng.random(G)
        eff = rng.normal(0, 1.5, G)[g]                       # building-SHAPE effect the model cannot see
        y = 10 * X[:, 0] + 5 * np.sin(4 * X[:, 1]) + 3 * X[:, 2] * X[:, 3] + eff + rng.normal(0, 1, len(g)) * (1 + X[:, 4])
        tr_g = rng.permutation(G)[:16]
        tr, te = np.isin(g, tr_g), ~np.isin(g, tr_g)
        mk = lambda: GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=0)
        q, _ = mc.group_conformal(mk, X[tr], y[tr], g[tr], alpha=0.1, n_splits=8)
        m = mk().fit(X[tr], y[tr])
        cov_c.append(np.mean(np.abs(y[te] - m.predict(X[te])) <= q))
        cov_n.append(np.mean(np.abs(y[te] - m.predict(X[te])) <= 1.645 * np.std(y[tr] - m.predict(X[tr]))))
    c, n = np.mean(cov_c), np.mean(cov_n)
    return c >= 0.86 and n < c - 0.05, f"target 90%: conformal coverage on UNSEEN shapes {c:.1%} vs naive in-sample-residual interval {n:.1%}"


def P2_shrinkage(reps=300):
    out = {}
    for tag, (a, b) in {"rare": (3, 3), "common": (4, 0.5)}.items():      # rare = ~1 severe event/yr; common = ~8 events/yr
        win, red, cov = [], [], []
        for r in range(reps):
            rng = np.random.default_rng(r)
            lam = rng.gamma(a, 1 / b, 49); e = rng.uniform(3, 10, 49)
            y = rng.poisson(lam * e)
            o = mc.eb_gamma_poisson(y, e)
            mr, me = np.mean((o["raw"] - lam) ** 2), np.mean((o["mean"] - lam) ** 2)
            win.append(me < mr); red.append(1 - me / mr); cov.append(np.mean((lam >= o["lo"]) & (lam <= o["hi"])))
        out[tag] = (np.mean(win), np.mean(red), np.mean(cov))
    ok = out["rare"][0] > 0.9 and out["rare"][1] > 0.2 and out["common"][1] > -0.02 and all(0.82 < v[2] < 0.96 for v in out.values())
    return ok, (f"RARE events (~1/yr): EB beats raw in {out['rare'][0]:.0%} of {reps} worlds, MSE -{out['rare'][1]:.0%}, 90% interval covers {out['rare'][2]:.1%} | "
                f"COMMON events (~8/yr): gain only {out['common'][1]:.0%} (noise is small, so little to borrow) and never materially worse, coverage {out['common'][2]:.1%}")


def P3_smaa():
    rng = np.random.default_rng(1)
    M = rng.random((5, 2)); ex = mc.smaa_exact_2crit(M)
    ns, errs = [500, 2000, 8000, 32000, 128000], []
    for n in ns:
        errs.append(np.mean([np.abs(mc.smaa(M, n, s)[:, 0] - ex).max() for s in range(25)]))
    slope = np.polyfit(np.log(ns), np.log(errs), 1)[0]
    return (-0.65 < slope < -0.35 and errs[-1] < 0.004), f"max error vs exact integral falls {errs[0]:.4f} -> {errs[-1]:.4f}; log-log slope {slope:.2f} (theory -0.5)"


def P4_cvar():
    rng = np.random.default_rng(2)
    L = rng.normal(0, 1, 1000)
    d1 = abs(mc.cvar_empirical(L, 0.9) - np.sort(L)[-100:].mean())
    J, N = 5, 600
    S = rng.normal(rng.uniform(-50, 300, J), rng.uniform(80, 250, J), (N, J))
    ab, co, cap, B = rng.uniform(100, 900, J), rng.uniform(2, 6, J), 60, 600
    r = mc.cvar_lp(ab, co, S, B, cap, 0.9)
    x = rng.random((200000, J)) * cap
    ok = (x @ co <= B)
    loss = -(x[ok] @ S.T)
    k = int(np.ceil(0.1 * N)); cv = np.sort(loss, axis=1)[:, -k:].mean(1)
    feas = cv <= 1e-9
    best_rand = (x[ok][feas] @ ab).max() if feas.any() else 0.0
    # slack tail constraint -> must equal fractional-knapsack greedy
    S_big = np.abs(S) + 1e3
    r2 = mc.cvar_lp(ab, co, S_big, B, cap, 0.9)
    order, left, gr = np.argsort(-ab / co), B, 0.0
    for j in order:
        take = min(cap, left / co[j]); gr += take * ab[j]; left -= take * co[j]
    fr = [mc.cvar_lp(ab, co, S - s, B, cap, 0.9)["obj"] for s in (0, 10, 25, 50)]   # demanding more worst-case saving
    return (d1 < 1e-9 and r["obj"] >= best_rand - 1e-6 and r["cvar"] <= 1e-6 and abs(r2["obj"] - gr) < 1e-5 * gr and fr[0] >= fr[1] >= fr[2] >= fr[3] and fr[2] > 0), \
        f"CVaR formula error {d1:.1e}; LP optimum {r['obj']:.0f} >= best of {int(feas.sum())} random feasible plans ({best_rand:.0f}); " \
        f"CVaR constraint met ({r['cvar']:.1e}); equals greedy knapsack when tail slack ({r2['obj']:.0f} vs {gr:.0f}); abatement falls {fr[0]:.0f} -> {fr[1]:.0f} -> {fr[2]:.0f} -> {fr[3]:.0f} as the required worst-case saving rises"


def P5_evt_trend(reps=300):
    xi, sg, u, lam, yrs, T = 0.1, 2.0, 30.0, 5.0, 30, 50
    true = float(mc.return_level(u, xi, sg, lam, T))
    cover, xis = [], []
    for r in range(reps):
        exc = stats.genpareto.rvs(xi, scale=sg, size=max(20, np.random.default_rng(r).poisson(lam * yrs)), random_state=r)
        o = mc.rl_bootstrap(exc, u, yrs, T, B=300, seed=r)
        cover.append(o["lo"] <= true <= o["hi"]); xis.append(o["xi"])
    # trend test: false-positive rate under AR(1) noise, no trend
    rng = np.random.default_rng(5); fp_n, fp_c, pw_c = 0, 0, 0
    R = 1200
    for _ in range(R):
        e = np.zeros(50); e[0] = rng.normal()
        for i in range(1, 50):
            e[i] = 0.6 * e[i - 1] + rng.normal()
        fp_n += mc.mk_test(e, False)["p"] < 0.05; fp_c += mc.mk_test(e, True)["p"] < 0.05
        pw_c += mc.mk_test(e + 0.12 * np.arange(50), True)["p"] < 0.05
    # Theil-Sen vs OLS under outliers
    eo, es = [], []
    for r in range(300):
        rr = np.random.default_rng(r); t = np.arange(40.0); y = 0.05 * t + rr.normal(0, .3, 40)
        k = rr.choice(40, 5, replace=False); y[k] += rr.choice([-6, 6], 5)
        eo.append(abs(np.polyfit(t, y, 1)[0] - 0.05)); es.append(abs(stats.theilslopes(y, t)[0] - 0.05))
    return (0.80 <= np.mean(cover) <= 0.97 and fp_c / R < 0.5 * fp_n / R and np.median(es) < 0.5 * np.median(eo)), \
        f"50-yr return level 90% CI covers truth {np.mean(cover):.1%}; xi_hat mean {np.mean(xis):.3f} (true 0.1) | " \
        f"MK false-positive rate under AR(1): naive {fp_n / R:.1%} -> corrected {fp_c / R:.1%} (power {pw_c / R:.0%}) | " \
        f"Theil-Sen median error {np.median(es):.4f} vs OLS {np.median(eo):.4f} with 12% outliers"


def P6_panel(reps=300):
    beta, res = 0.5, {"fe": [], "ols": [], "cov_cl": [], "cov_iid": []}
    for r in range(reps):
        rng = np.random.default_rng(r); G, T = 40, 15
        m = rng.normal(0, 3, G); a = -1.2 * m + rng.normal(0, 1, G)           # hot places have lower baseline yield
        e, xx = np.zeros((G, T)), np.zeros((G, T))                       # temperatures AND shocks are persistent
        e[:, 0], xx[:, 0] = rng.normal(0, 1, G), rng.normal(0, 1, G)
        for t in range(1, T):
            e[:, t] = 0.6 * e[:, t - 1] + rng.normal(0, 1, G); xx[:, t] = 0.7 * xx[:, t - 1] + rng.normal(0, 1, G)
        x = m[:, None] + xx
        y = beta * x + a[:, None] + e
        g = np.repeat(np.arange(G), T)
        o = mc.fe_cluster(y.ravel(), x.ravel(), g)
        ols = np.polyfit(x.ravel(), y.ravel(), 1)[0]
        res["fe"].append(o["b"][0] - beta); res["ols"].append(ols - beta)
        res["cov_cl"].append(abs(o["b"][0] - beta) <= stats.t.ppf(.975, G - 1) * o["se"][0]); res["cov_iid"].append(abs(o["b"][0] - beta) <= 1.96 * o["se_iid"][0])
    return (abs(np.mean(res["fe"])) < 0.02 and abs(np.mean(res["ols"])) > 0.3 and np.mean(res["cov_cl"]) > 0.92 > np.mean(res["cov_iid"])), \
        f"true effect 0.50: pooled OLS bias {np.mean(res['ols']):+.2f}, fixed-effects bias {np.mean(res['fe']):+.3f}; 95% CI coverage cluster-robust (t, G-1 df) {np.mean(res['cov_cl']):.1%} vs naive {np.mean(res['cov_iid']):.1%}"


def P7_sobol():
    f = lambda X: np.sin(X[:, 0]) + 7 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])
    S1, ST = mc.sobol_indices(f, [(-np.pi, np.pi)] * 3, n=2 ** 15)
    e1, eT = np.array([0.3139, 0.4424, 0.0]), np.array([0.5576, 0.4424, 0.2437])    # analytic Ishigami (a=7,b=0.1)
    dm = max(np.abs(S1 - e1).max(), np.abs(ST - eT).max())
    o, m_in, m_dec, m_acc = mc.fod_vec(1000, 0.02, 0.5, 0.0, years=range(2011, 2400), mass_check=True)
    cons = abs((m_dec + m_acc) / m_in - 1)
    A = np.array([4.0, 2.0, 1.0, 3.0]); Pm = np.random.default_rng(0).dirichlet(A, 800)
    ah = mc.dirichlet_mle(Pm); dre = np.max(np.abs(ah - A) / A)
    return (dm < 0.03 and cons < 1e-9 and dre < 0.12), \
        f"Sobol vs analytic Ishigami max error {dm:.3f}; landfill FOD mass balance error {cons:.1e}; Dirichlet MLE max relative error {dre:.1%}"


def P8_circular():
    c, s, y = 0.8, 0.7, 0.9
    th, sim = mc.chain_stats(c, s, y)["E_uses"], mc.chain_sim(c, s, y)
    rows = []
    for rho, g in [(0.5, 0.0), (0.5, 0.03), (0.4, 0.05)]:
        rows.append((mc.cmu_sim(rho, g, 10, 2, seed=1), float(mc.cmu_gamma(rho, g, 10, 2))))
    err = max(abs(a - b) / b for a, b in rows)
    return (abs(th - sim) / th < 0.01 and err < 0.04), \
        f"expected uses per tonne: theory {th:.3f} vs {200000:,}-parcel simulation {sim:.3f}; stock-flow CMU simulation vs Gamma closed form worst relative error {err:.1%}"


def P9_scurve(n=3000):
    """Regime: shares inside (0.5%, 99.5%) over the fitting window (the app refuses saturated countries).
    Each country gets its own target (predicted level +/- 2 s.e.) so calibration is tested across the WHOLE 0-1 range."""
    rng = np.random.default_rng(9); t = np.arange(2000, 2021.0)
    pit, p, hit, cov = [], [], [], []
    for _ in range(n):
        k, t0, sd = rng.uniform(.05, .2), rng.uniform(2005, 2025), rng.uniform(.04, .15)
        z = k * (t - t0) + rng.normal(0, sd, len(t)); s = 100 / (1 + np.exp(-z))
        zf = k * (2035 - t0) + rng.normal(0, sd)
        o0 = mc.logistic_forecast(t, s, 2035, 50.0)
        ztar = o0["m"] + rng.uniform(-2, 2) * o0["se"]
        o = mc.logistic_forecast(t, s, 2035, 100 / (1 + np.exp(-ztar)))
        pit.append(stats.t.cdf((zf - o["m"]) / o["se"], o["df"])); p.append(o["p_hit"]); hit.append(zf >= ztar)
        cov.append(o["lo"] <= 100 / (1 + np.exp(-zf)) <= o["hi"])
    p, hit = np.array(p), np.array(hit, float)
    brier, base = np.mean((p - hit) ** 2), np.mean((hit.mean() - hit) ** 2)
    bins = np.digitize(p, [.1, .3, .5, .7, .9]); gap = max(abs(p[bins == b].mean() - hit[bins == b].mean()) for b in np.unique(bins) if (bins == b).sum() > 40)
    ks = stats.kstest(pit, "uniform").pvalue
    return (ks > 0.01 and abs(np.mean(cov) - 0.9) < 0.03 and gap < 0.07 and abs(brier - np.mean(p * (1 - p))) < 0.01), \
        f"{n} simulated countries: PIT uniformity KS p={ks:.2f}; 90% interval coverage {np.mean(cov):.1%}; worst reliability gap {gap:.3f}; Brier {brier:.3f} vs {np.mean(p * (1 - p)):.3f} expected if perfectly calibrated (climatology {base:.3f})"


def P10_bandit():
    p = [0.30, 0.34, 0.38, 0.45]
    out = {k: mc.bandit_regret(p, 3000, 200, k, seed=3) for k in ("ts", "uniform", "eps")}
    ts, un, ep = out["ts"][0], out["uniform"][0], out["eps"][0]
    return (ts[-1] < 0.25 * un[-1] and ts[-1] < ep[-1] and ts[-1] / ts[299] < 0.5 * un[-1] / un[299] and out["ts"][1][3] > 0.8), \
        f"cumulative regret after 3000 households: Thompson {ts[-1]:.0f} vs equal-split A/B {un[-1]:.0f} vs eps-greedy {ep[-1]:.0f}; " \
        f"share given the best message {out['ts'][1][3]:.0%}; 10x more households -> regret x{ts[-1] / ts[299]:.1f} (A/B split: x{un[-1] / un[299]:.1f})"


TESTS = [("1 Conformal intervals (buildings)", "Coverage holds on building shapes the model has never seen", P1_conformal),
         ("2 Empirical-Bayes shrinkage (heatwaves)", "Pooling beats raw counts and ranks come with honest probabilities", P2_shrinkage),
         ("3 SMAA (triage weights)", "Rank acceptability converges to the exact answer at the 1/sqrt(n) rate", P3_smaa),
         ("4 CVaR linear programme (retrofit budget)", "LP is optimal, tail-risk-safe, and collapses to greedy knapsack when risk is slack", P4_cvar),
         ("5 Extreme values + trend (resilience)", "GPD return-level CIs are honest; autocorrelation-corrected trend test controls false alarms", P5_evt_trend),
         ("6 Fixed-effects panel (farming)", "Pooled OLS is biased by place effects; FE + cluster SEs recover the truth", P6_panel),
         ("7 Sobol + mass-balanced landfill model (waste)", "Sensitivity indices match the analytic benchmark; no methane mass created or lost", P7_sobol),
         ("8 Markov + Gamma-lifetime circularity (industry)", "Closed-form circular-use rate matches item-level simulation", P8_circular),
         ("9 Logistic S-curve forecast (electrification)", "Probability of hitting 35% by 2035 is calibrated", P9_scurve),
         ("10 Thompson sampling (awareness)", "Learns the best message with logarithmic regret, far better than a fixed A/B split", P10_bandit)]


def run_all(verbose=False):
    rows = []
    for name, claim, fn in TESTS:
        t0 = time.time()
        try:
            ok, ev = fn()
        except Exception as ex:                                  # a crashing proof counts as a failed proof
            ok, ev = False, f"ERROR {type(ex).__name__}: {ex}"
        rows.append(dict(Method=name, Claim=claim, Result="PASS" if ok else "FAIL", Evidence=ev, Seconds=round(time.time() - t0, 1)))
        if verbose:
            print(f"[{rows[-1]['Result']}] {name}  ({rows[-1]['Seconds']}s)\n      {claim}\n      {ev}\n", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_all(verbose=True)
    print(f"{(df.Result == 'PASS').sum()}/{len(df)} proofs passed")
    sys.exit(0 if (df.Result == "PASS").all() else 1)
