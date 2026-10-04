"""
mathcore.py - the 10 new mathematical methods behind HeatShift 2035 (pure NumPy/SciPy, no Streamlit).
Every function here is validated against a KNOWN ground truth in proofs.py.

 1 group_conformal      distribution-free prediction intervals for UNSEEN building shapes
 2 eb_gamma_poisson     empirical-Bayes shrinkage + posterior rank probabilities (noisy heatwave counts)
 3 smaa                 Stochastic Multicriteria Acceptability Analysis (triage robust to ANY weights)
 4 cvar_lp              Rockafellar-Uryasev CVaR linear programme (retrofit budget with a tail-risk guarantee)
 5 gpd_pwm/return_level extreme-value (GPD) return levels + Theil-Sen/Mann-Kendall (autocorrelation-corrected)
 6 fe_cluster           fixed-effects panel regression, cluster-robust SEs (climate -> crop yield)
 7 sobol_indices/fod_vec Sobol global sensitivity of landfill methane (mass-conserving IPCC FOD) + Dirichlet MLE
 8 chain_stats/cmu_*    absorbing-Markov-chain + Gamma-lifetime model of circular material use
 9 logistic_forecast    logit-OLS S-curve with exact t-predictive probability of hitting the 35% target
10 bandit_regret        Thompson sampling to pick the awareness message that actually works
"""
import numpy as np
import scipy.sparse as sp
from scipy import optimize, special, stats
from scipy.optimize import linprog
from sklearn.model_selection import GroupKFold


# ============================================================ 1. conformal prediction for unseen groups
def group_conformal(make_model, X, y, groups, alpha=0.1, n_splits=6):
    """Leave-whole-groups-out out-of-fold residuals -> finite-sample conformal quantile (CV+ style).
    Interval for a NEW group's point: prediction +/- q.  Valid when groups are exchangeable."""
    X, y, g = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    oof = np.empty(len(y))
    for tr, te in GroupKFold(min(n_splits, len(np.unique(g)))).split(X, y, g):
        oof[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
    r, n = np.abs(y - oof), len(y)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(r, lvl, method="higher")), oof


# ============================================================ 2. empirical Bayes (Gamma-Poisson)
def eb_gamma_poisson(y, e, level=0.9):
    """y_i ~ Poisson(lam_i * e_i), lam_i ~ Gamma(a, b).  (a, b) by marginal (negative-binomial) MLE."""
    y, e = np.asarray(y, float), np.asarray(e, float)

    def nll(p):
        a, b = np.exp(np.clip(p, -8, 12))
        return -np.sum(special.gammaln(a + y) - special.gammaln(a) - special.gammaln(y + 1)
                       + a * np.log(b / (b + e)) + y * np.log(e / (b + e)))
    m = max(y.sum() / e.sum(), 1e-6)
    r = optimize.minimize(nll, [np.log(2.0), np.log(2.0 / m)], method="Nelder-Mead",
                          options=dict(xatol=1e-6, fatol=1e-9, maxiter=3000))
    a, b = np.exp(np.clip(r.x, -8, 12))
    pa, pb = a + y, b + e
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return dict(a=a, b=b, pa=pa, pb=pb, mean=pa / pb, raw=y / e,
                lo=stats.gamma.ppf(lo, pa, scale=1 / pb), hi=stats.gamma.ppf(hi, pa, scale=1 / pb))


def posterior_ranks(pa, pb, n=20000, seed=0):
    """Draw from the Gamma posteriors; return (n x places) matrix of ranks (0 = highest rate)."""
    D = np.random.default_rng(seed).gamma(pa, 1 / pb, size=(n, len(pa)))
    return (-D).argsort(1).argsort(1)


# ============================================================ 3. SMAA
def smaa(M, n=50000, seed=0):
    """M: alternatives x criteria (higher = better). Weights ~ Dirichlet(1,..,1) (i.e. ANY weights, equally likely).
    Returns the rank-acceptability matrix RAI[alternative, rank] (rank 0 = best)."""
    M = np.asarray(M, float)
    A = M.shape[0]
    W = np.random.default_rng(seed).dirichlet(np.ones(M.shape[1]), n)
    order = np.argsort(-(W @ M.T), axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.tile(np.arange(A), (n, 1)), axis=1)
    return np.stack([(ranks == r).mean(0) for r in range(A)], 1)


def smaa_exact_2crit(M, grid=400000):
    """Exact (deterministic-grid) probability that each alternative is best when w ~ U(0,1) on 2 criteria."""
    M = np.asarray(M, float)
    w = (np.arange(grid) + 0.5) / grid
    best = np.argmax(np.outer(w, M[:, 0]) + np.outer(1 - w, M[:, 1]), axis=1)
    return np.bincount(best, minlength=len(M)) / grid


# ============================================================ 4. CVaR linear programme
def cvar_empirical(loss, alpha=0.9):
    loss = np.sort(np.asarray(loss, float))[::-1]
    k = int(np.ceil((1 - alpha) * len(loss) - 1e-12))
    return float(loss[:max(k, 1)].mean())


def cvar_lp(abate, cost, S, budget, cap, alpha=0.9):
    """maximise abate.x  s.t.  cost.x <= budget,  0 <= x <= cap,  CVaR_alpha(-S x) <= 0.
    S: scenarios x options (annual $ saving per retrofit).  Rockafellar-Uryasev (2000) linearisation."""
    S = np.asarray(S, float)
    N, J = S.shape
    cap = np.broadcast_to(np.asarray(cap, float), (J,))
    c = np.r_[-np.asarray(abate, float), 0.0, np.zeros(N)]
    A = sp.vstack([sp.csr_matrix(np.r_[np.asarray(cost, float), 0.0, np.zeros(N)][None, :]),
                   sp.csr_matrix(np.r_[np.zeros(J), 1.0, np.full(N, 1 / ((1 - alpha) * N))][None, :]),
                   sp.hstack([sp.csr_matrix(-S), sp.csr_matrix(-np.ones((N, 1))), -sp.identity(N)])], format="csr")
    b = np.r_[budget, 0.0, np.zeros(N)]
    bounds = [(0, float(v)) for v in cap] + [(None, None)] + [(0, None)] * N
    r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if r.status != 0:
        return None
    x = r.x[:J]
    return dict(x=x, obj=-r.fun, cvar=cvar_empirical(-(S @ x), alpha), spend=float(np.dot(cost, x)),
                exp_saving=float((S @ x).mean()), p10_saving=float(np.percentile(S @ x, 10)))


# ============================================================ 5. extreme values + trends
def decluster(x, u, gap=2):
    """Cluster maxima of exceedances over u (consecutive exceedances <= `gap` days apart form one event)."""
    x = np.asarray(x, float)
    idx = np.flatnonzero(x > u)
    if idx.size == 0:
        return np.array([])
    return np.array([x[g].max() for g in np.split(idx, np.flatnonzero(np.diff(idx) > gap) + 1)])


def gpd_pwm(exc):
    """Hosking-Wallis probability-weighted-moment GPD fit. Works on the last axis. Returns (xi, sigma)."""
    x = np.sort(np.asarray(exc, float), axis=-1)
    n = x.shape[-1]
    p = (np.arange(1, n + 1) - 0.35) / n
    a0, a1 = x.mean(-1), ((1 - p) * x).mean(-1)
    den = a0 - 2 * a1
    return np.clip(-(a0 / den - 2), -0.5, 0.5), 2 * a0 * a1 / den


def return_level(u, xi, sig, lam, T):
    xi = np.asarray(xi, float)
    with np.errstate(all="ignore"):
        return np.where(np.abs(xi) < 1e-6, u + sig * np.log(lam * T), u + sig / xi * ((lam * T) ** xi - 1))


def rl_bootstrap(exc, u, years, T=50, B=400, seed=0, level=0.9):
    exc = np.asarray(exc, float)
    n, rng = len(exc), np.random.default_rng(seed)
    xi, sig = gpd_pwm(exc)
    pt = float(return_level(u, xi, sig, n / years, T))
    xb, sb = gpd_pwm(exc[rng.integers(0, n, (B, n))])
    rl = return_level(u, xb, sb, np.maximum(rng.poisson(n, B), 1) / years, T)
    return dict(xi=float(xi), sigma=float(sig), rl=pt, lo=float(np.quantile(rl, (1 - level) / 2)),
                hi=float(np.quantile(rl, 1 - (1 - level) / 2)), lam=n / years)


def mk_test(x, correct=True):
    """Mann-Kendall trend test; `correct` inflates Var(S) by (1+rho)/(1-rho) for lag-1 autocorrelation."""
    x = np.asarray(x, float)
    n, t = len(x), np.arange(len(x))
    S = np.sign(x[None, :] - x[:, None])[np.triu_indices(n, 1)].sum()
    var, infl = n * (n - 1) * (2 * n + 5) / 18, 1.0
    if correct:
        res = x - stats.theilslopes(x, t)[0] * t
        res = res - np.median(res)
        rho = np.corrcoef(res[:-1], res[1:])[0, 1]
        rho = 0.0 if np.isnan(rho) else float(np.clip(rho, 0, 0.95))
        infl = (1 + rho) / (1 - rho)
    z = (S - np.sign(S)) / np.sqrt(var * infl)
    return dict(S=float(S), z=float(z), p=float(2 * (1 - stats.norm.cdf(abs(z)))), infl=infl)


# ============================================================ 6. fixed-effects panel regression
def fe_cluster(y, X, g):
    """Within estimator with cluster-robust (CR1) and naive iid standard errors."""
    y, X = np.asarray(y, float), np.asarray(X, float).reshape(len(y), -1)
    codes = np.unique(np.asarray(g), return_inverse=True)[1].ravel()
    G, cnt = codes.max() + 1, np.bincount(codes)
    dm = lambda v: v - (np.bincount(codes, weights=v) / cnt)[codes]
    yd, Xd = dm(y), np.column_stack([dm(X[:, j]) for j in range(X.shape[1])])
    N, K = Xd.shape
    inv = np.linalg.pinv(Xd.T @ Xd)
    b = inv @ (Xd.T @ yd)
    u = yd - Xd @ b
    sc = np.zeros((G, K))
    np.add.at(sc, codes, Xd * u[:, None])
    V = G / (G - 1) * (N - 1) / (N - K) * inv @ (sc.T @ sc) @ inv
    V0 = (u @ u) / (N - G - K) * inv
    return dict(b=b, se=np.sqrt(np.diag(V)), se_iid=np.sqrt(np.diag(V0)), V=V, G=G, N=N)


# ============================================================ 7. Sobol sensitivity + landfill FOD + Dirichlet
def sobol_indices(f, bounds, n=16384, seed=0):
    """Saltelli (2010) first-order and Jansen total-order Sobol indices."""
    rng = np.random.default_rng(seed)
    lo, hi = np.array([b[0] for b in bounds], float), np.array([b[1] for b in bounds], float)
    A, B = lo + (hi - lo) * rng.random((n, len(lo))), lo + (hi - lo) * rng.random((n, len(lo)))
    fA, fB = f(A), f(B)
    V = np.var(np.r_[fA, fB])
    S1, ST = np.empty(len(lo)), np.empty(len(lo))
    for i in range(len(lo)):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fAB = f(ABi)
        S1[i], ST[i] = np.mean(fB * (fAB - fA)) / V, 0.5 * np.mean((fA - fAB) ** 2) / V
    return S1, ST


FOD = {"Food": (0.40, 0.15, 0.185), "Garden": (0.15, 0.20, 0.10), "Paper": (0.15, 0.40, 0.06),
       "Wood": (0.05, 0.43, 0.03), "Textiles": (0.05, 0.24, 0.06)}  # share, DOC, k  (IPCC 2006 temperate-wet defaults)


def fod_vec(tonnes=60000, growth=0.03, divert=0.0, capture=0.0, gwp=28, growth_after=None, k_mult=1.0,
            doc_mult=1.0, docf=0.5, mcf=1.0, food_share=None, F=0.5, years=range(2011, 2036), delay=0.5,
            mass_check=False):
    """Vectorised, MASS-CONSERVING first-order decay.  Every argument may be an array (broadcast).
    acc_t = acc_{t-1} e^-k + dd_t e^(-k*delay);  dec_t = acc_{t-1}(1-e^-k) + dd_t (1-e^(-k*delay))
    => acc_t + dec_t = acc_{t-1} + dd_t exactly (what is buried is either decayed or still in the pile)."""
    ga = growth if growth_after is None else growth_after
    fs = FOD["Food"][0] if food_share is None else food_share
    tonnes, growth, ga, divert, capture, k_mult, doc_mult, docf, mcf, F, fs = np.broadcast_arrays(
        *[np.atleast_1d(np.asarray(v, float)) for v in (tonnes, growth, ga, divert, capture, k_mult, doc_mult, docf, mcf, F, fs)])
    out, m_in, m_dec, m_acc = (np.zeros((len(growth), len(years))), 0.0, 0.0, 0.0)
    for name, (frac, doc, k0) in FOD.items():
        k, acc = k0 * k_mult, np.zeros_like(growth)
        fr = fs if name == "Food" else frac
        for i, y in enumerate(years):
            ramp = float(np.clip((y - 2025) / 10, 0, 1))
            w = tonnes * (1 + growth) ** (min(y, 2026) - 2026) * (1 + ga) ** max(y - 2026, 0)
            share = fr * (1 - divert * ramp) if name in ("Food", "Garden") else fr
            dd = w * share * doc * doc_mult * docf * mcf
            dec = acc * (1 - np.exp(-k)) + dd * (1 - np.exp(-k * delay))
            acc = acc * np.exp(-k) + dd * np.exp(-k * delay)
            out[:, i] += dec * F * 16 / 12 * (1 - capture * ramp) * gwp
            m_in, m_dec = m_in + dd.sum(), m_dec + dec.sum()
        m_acc += acc.sum()
    return (out, m_in, m_dec, m_acc) if mass_check else out


def _inv_digamma(y, iters=8):
    x = np.where(y >= -2.22, np.exp(y) + 0.5, -1 / (y - special.digamma(1)))
    for _ in range(iters):
        x = x - (special.digamma(x) - y) / special.polygamma(1, x)
    return x


def dirichlet_mle(P, iters=500):
    """Minka fixed-point MLE for a Dirichlet from compositional rows (rows are renormalised)."""
    P = np.clip(np.asarray(P, float), 1e-6, None)
    P = P / P.sum(1, keepdims=True)
    lp, m, v = np.log(P).mean(0), P.mean(0), P.var(0) + 1e-12
    s = float(np.clip(np.median(m * (1 - m) / v - 1), 1.0, 1e4))
    a = m * s
    for _ in range(iters):
        new = _inv_digamma(special.digamma(a.sum()) + lp)
        if np.max(np.abs(new - a)) < 1e-9:
            a = new
            break
        a = new
    return a


# ============================================================ 8. circular materials: Markov chain + Gamma lifetimes
def chain_stats(c, s, y):
    """States: 0 in-use -> (c) 1 collected -> (s) 2 sorted -> (y) back to 0 (reprocessed). Else lost."""
    Q = np.array([[0, c, 0], [0, 0, s], [y, 0, 0]], float)
    N = np.linalg.inv(np.eye(3) - Q)
    return dict(E_uses=float(N[0, 0]), rho=float(c * s * y), virgin_per_use=float(1 - c * s * y), N=N)


def chain_sim(c, s, y, n=200000, seed=0):
    rng, P = np.random.default_rng(seed), np.array([c, s, y])
    state, uses, alive = np.zeros(n, int), np.ones(n), np.ones(n, bool)
    while alive.any():
        idx = np.flatnonzero(alive)
        go = rng.random(idx.size) < P[state[idx]]
        alive[idx[~go]] = False
        mv = idx[go]
        state[mv] = (state[mv] + 1) % 3
        uses[mv[state[mv] == 0]] += 1
    return float(uses.mean())


def cmu_gamma(rho, g, mean_life=10.0, shape=2.0):
    """Closed form: CMU = rho * E[(1+g)^-L], L ~ Gamma(shape, mean_life/shape)  =  rho (1 + theta ln(1+g))^-shape."""
    return rho * (1 + mean_life / shape * np.log1p(g)) ** (-shape)


def cmu_sim(rho, g, mean_life=10.0, shape=2.0, T=100, D0=5000, seed=0):
    """Item-level stochastic stock-flow simulation: growing demand, Gamma lifetimes, each retired item recycled
    with probability rho and re-enters use (so recycled items recycle again). Returns mean CMU over last 10 years."""
    rng = np.random.default_rng(seed)
    retire, cm = np.zeros(T + 400), []
    for t in range(T):
        D = int(round(D0 * (1 + g) ** t))
        rec = min(int(rng.binomial(int(retire[t]), rho)), D)
        life = np.maximum(1, np.rint(rng.gamma(shape, mean_life / shape, D))).astype(int)
        retire += np.bincount(t + life, minlength=len(retire))[:len(retire)]
        cm.append(rec / D)
    return float(np.mean(cm[-10:]))


# ============================================================ 9. logistic S-curve with exact predictive probability
def logistic_forecast(t, s, t_new, target, L=100.0):
    """logit(s/L) = b0 + b1 t + e.  Flat-prior Bayesian posterior predictive == OLS t-prediction interval (exact)."""
    t, s = np.asarray(t, float), np.clip(np.asarray(s, float), 0.5, L - 0.5)
    z = np.log(s / (L - s))
    X = np.column_stack([np.ones_like(t), t])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ z
    df = len(t) - 2
    s2 = float(((z - X @ b) ** 2).sum() / df)
    x0 = np.array([1.0, t_new])
    m, se = float(x0 @ b), float(np.sqrt(s2 * (1 + x0 @ XtXi @ x0)))
    q = stats.t.ppf([0.05, 0.95], df)
    zt = np.log(target / (L - target))
    ex = lambda v: L / (1 + np.exp(-v))
    return dict(mean=ex(m), lo=ex(m + q[0] * se), hi=ex(m + q[1] * se), p_hit=float(1 - stats.t.cdf((zt - m) / se, df)),
                m=m, se=se, df=df, slope=float(b[1]), t_mid=float(-b[0] / b[1]) if b[1] else np.nan)


# ============================================================ 10. Thompson sampling
def bandit_regret(p, T=3000, R=200, policy="ts", seed=0, eps=0.1):
    rng, p = np.random.default_rng(seed), np.asarray(p, float)
    K, ar = len(p), np.arange(R)
    a, b, pulls, reg = np.ones((R, K)), np.ones((R, K)), np.zeros((R, K)), np.zeros((R, T))
    for t in range(T):
        if policy == "ts":
            arm = rng.beta(a, b).argmax(1)
        elif policy == "uniform":
            arm = rng.integers(0, K, R)
        else:
            arm = np.where(rng.random(R) < eps, rng.integers(0, K, R), (a / (a + b)).argmax(1))
        r = (rng.random(R) < p[arm]).astype(float)
        a[ar, arm] += r
        b[ar, arm] += 1 - r
        pulls[ar, arm] += 1
        reg[:, t] = p.max() - p[arm]
    return np.cumsum(reg, 1).mean(0), pulls.mean(0) / T
