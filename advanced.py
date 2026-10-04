"""
advanced.py - Streamlit tabs for the 10 new mathematical methods (see mathcore.py) and 6 NEW Kaggle datasets.
Datasets are found by their COLUMNS (drop the CSVs in this folder or ./data). Every panel degrades gracefully.
"""
import os, re
import numpy as np, pandas as pd, streamlit as st, plotly.express as px, plotly.graph_objects as go
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
import mathcore as mc
import addons as ad
import proofs

PACIFIC = ["Australia", "New Zealand", "Fiji", "Samoa", "Tonga", "Vanuatu", "Solomon Islands", "Papua New Guinea", "Kiribati", "Tuvalu", "Micronesia (country)"]
NEW_DATA = [("Global Data on Sustainable Energy (2000-2020)", "anshtanwar/global-data-on-sustainable-energy", ("accesstoelectricity", "entity")),
            ("World Energy Consumption (Our World in Data)", "pralabhpoudel/world-energy-consumption", ("isocode", "primaryenergyconsumption")),
            ("What a Waste Global Dataset (World Bank)", "mannmann2/what-a-waste-global-dataset", ("compositionfood",)),
            ("Climate Change: Earth Surface Temperature (Berkeley Earth)", "berkeleyearth/climate-change-earth-surface-temperature-data", ("averagetemperature", "city", "country")),
            ("Crop Yield Prediction (FAO + World Bank)", "patelris/crop-yield-prediction-dataset", ("hghayield", "avgtemp", "pesticides")),
            ("Twitter Climate Change Sentiment", "edqian/twitter-climate-change-sentiment-dataset", ("sentiment", "message"))]


def col(d, frag):
    return next((c for c in d.columns if frag in c), None)


def status():
    return pd.DataFrame([dict(dataset=n, kaggle=f"kaggle.com/datasets/{s}", found=(ad.by_col(*f) or "NOT FOUND")) for n, s, f in NEW_DATA])


def fmt_p(p):
    return f"{p:.0%}"


# ============================================================ 1. ELECTRIFICATION: S-curve + CVaR retrofit budget
@st.cache_data
def load_sust(p):
    d = ad.read(p)
    return d.rename(columns={col(d, "access_to_electricity"): "access"})[["entity", "year", "access"]].dropna()


@st.cache_data
def load_owid(p):
    d = ad.read(p)
    if "electricity_share_energy" in d.columns:
        d["share"] = d["electricity_share_energy"]
    elif "electricity_demand" in d.columns:
        d["share"] = 100 * d["electricity_demand"] / d["primary_energy_consumption"]
    else:
        return None
    return d[["country", "year", "share"]].dropna().query("share > 0")


def fit_row(g, target, yr, L, tmin):
    g = g[(g.year >= tmin) & (g.v > 0.5) & (g.v < L - 0.5)].sort_values("year")
    if len(g) < 8 or g.v.iloc[-1] <= g.v.iloc[0]:
        return None
    return mc.logistic_forecast(g.year.values, g.v.values, yr, target, L)


def scenarios(T, scops, deltas, elec, gas, N=400, seed=3):
    r = np.random.default_rng(seed)
    e, g, ef, lm, dl = elec * r.lognormal(0, .2, N), gas * r.lognormal(0, .25, N), r.uniform(.7, .9, N), r.lognormal(0, .3, N), r.normal(0, .3, N)
    S = np.empty((N, len(T)))
    for j, row in enumerate(T.itertuples()):
        sc, dem = np.interp(dl, deltas, scops[row.Location]), row.demand * lm
        S[:, j] = dem / ef * g - dem / sc * e
    return S


def electrification_tab(tab, X):
    with tab:
        st.subheader("Electrification maths: S-curve forecast + tail-risk-safe retrofit budget")
        sp_, op = ad.by_col("accesstoelectricity", "entity"), ad.by_col("isocode", "primaryenergyconsumption")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**A. Will electricity access reach the target? (Pacific lens)** - method 9, logit-OLS with exact t-predictive probability")
            if sp_:
                d = load_sust(sp_).rename(columns={"access": "v"})
                yr, tgt = st.slider("Target year", 2025, 2040, 2030), st.slider("Access target, % of population", 80.0, 99.9, 99.0, 0.5)
                rows = []
                for ent, g in d.groupby("entity"):
                    last = g.sort_values("year").v.iloc[-1]
                    o = fit_row(g, tgt, yr, 100.0, 2000)
                    rows.append(dict(Country=ent, Latest=last, Forecast=(o or {}).get("mean", np.nan), Lo90=(o or {}).get("lo", np.nan),
                                     Hi90=(o or {}).get("hi", np.nan), P_hit=1.0 if last >= tgt else (o or {}).get("p_hit", np.nan),
                                     Note="already there" if last >= tgt else ("" if o else "saturated/flat - model not applicable")))
                R = pd.DataFrame(rows)
                pac = R[R.Country.isin(PACIFIC)].sort_values("P_hit")
                st.dataframe(pac.round(2), hide_index=True, use_container_width=True)
                gaps = R[(R.Latest < tgt) & R.P_hit.notna()].sort_values("P_hit").head(10)
                st.caption("Ten countries furthest from the target (lowest probability of getting there on current trend):")
                st.dataframe(gaps.round(2), hide_index=True, use_container_width=True)
                bt = []                                    # real-data backtest: fit 2000-2012, predict 2020
                for ent, g in d.groupby("entity"):
                    a, b = g[g.year <= 2012], g[g.year == 2020]
                    o = fit_row(a, 50, 2020, 100.0, 2000) if len(b) else None
                    if o:
                        bt.append(o["lo"] <= b.v.iloc[0] <= o["hi"])
                if bt:
                    st.metric("Backtest on REAL data: 90% interval (fit 2000-12) contains the 2020 value", f"{np.mean(bt):.0%}", f"{len(bt)} countries")
            else:
                ad.missing("Global Data on Sustainable Energy (2000-2020)", "anshtanwar/global-data-on-sustainable-energy")
        with c2:
            st.markdown("**B. Probability a country reaches the global goal: 35% electrified by 2035**")
            if op and (o := load_owid(op)) is not None:
                L = st.slider("Realistic ceiling for electricity share, %", 40, 90, 60, 5)
                yrs = st.slider("Fit window start", 1990, 2010, 2000)
                ctry = sorted(o.country.unique())
                pick = st.multiselect("Countries", ctry, default=[c for c in ["Australia", "New Zealand", "Fiji", "World"] if c in ctry])
                rows = []
                for cn in pick:
                    r = fit_row(o[o.country == cn].rename(columns={"share": "v"}), 35.0, 2035, float(L), yrs)
                    last = o[o.country == cn].sort_values("year").share.iloc[-1]
                    rows.append(dict(Country=cn, Latest=last, Forecast2035=(r or {}).get("mean", np.nan), Lo90=(r or {}).get("lo", np.nan),
                                     Hi90=(r or {}).get("hi", np.nan), P_reach_35=(r or {}).get("p_hit", np.nan)))
                st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, use_container_width=True)
                st.caption("Proxy: electricity's share of energy in the OWID table (not exactly 'final-energy electrification'). "
                           "Logistic form assumed - the backtest on the left is the check.")
            else:
                ad.missing("World Energy Consumption (OWID)", "pralabhpoudel/world-energy-consumption")
        st.markdown("---\n**C. Where should a fixed retrofit budget go?** - method 4, CVaR linear programme (Rockafellar-Uryasev)")
        T, k = X["T"], st.columns(4)
        bud, cap = k[0].slider("Budget, $ million", 1, 200, 20), k[1].slider("Max homes per place", 100, 20000, 2000, 100)
        al = k[2].slider("Tail level alpha (worst 1-alpha of futures)", 0.80, 0.99, 0.90, 0.01)
        S = scenarios(T, X["scops"], X["DELTAS"], X["elec"], X["gas"])
        abate, cost = T.co2_35.values.clip(min=0), np.full(len(T), float(X["capex"]))
        nt = max(1, int(round((1 - al) * len(S))))
        order, left, xn = np.argsort(-T["saving"].values), bud * 1e6, np.zeros(len(T))      # naive: best expected saving first
        for j in order:
            xn[j] = min(cap, left / cost[j]); left -= xn[j] * cost[j]
        tail_n = np.sort(S @ xn)[:nt].mean() / max(cost @ xn / 1000, 1e-9)               # naive plan's worst-case saving per $1000
        req = k[3].slider("Guaranteed worst-case saving, $/yr per $1000 spent", -20, 60, int(np.clip(round(0.7 * tail_n), -20, 60)),
                          help="Default = 70% of what the 'best savings first' plan achieves in its worst futures.")

        def solve(r):
            return mc.cvar_lp(abate, cost, S - r / 1000 * cost, bud * 1e6, cap, al)
        sol = solve(req)
        if sol is None:
            st.warning("No feasible plan with that guarantee - lower the guaranteed saving.")
            return
        x = sol["x"]
        if x.sum() < 1:
            st.warning(f"No home can meet a worst-case saving of ${req}/yr per $1000 under these assumptions (the 'best savings first' plan reaches ${tail_n:,.0f}). Lower the guarantee.")
        tail_x = np.sort(S @ x)[:nt].mean() / max(cost @ x / 1000, 1e-9)
        m = st.columns(4)
        m[0].metric("CO2e avoided in 2035 (CVaR plan)", f"{abate @ x / 1000:,.0f} t/yr", f"{(abate @ x - abate @ xn) / 1000:+,.0f} vs 'best savings first'")
        m[1].metric("Homes retrofitted", f"{x.sum():,.0f}", f"naive plan {xn.sum():,.0f}")
        m[2].metric("Worst-case saving per $1000 (CVaR plan)", f"${tail_x:,.0f}/yr", f"naive plan ${tail_n:,.0f}/yr")
        m[3].metric("Expected saving", f"${sol['exp_saving'] / 1e6:,.1f}m/yr")
        st.caption("The guarantee holds for the CVaR plan by construction (average of the worst futures >= the slider). "
                   + ("The naive plan breaks it." if tail_n < req - 1e-6 else "The naive plan happens to meet it here."))
        plan = pd.DataFrame(dict(Place=T.Location, Homes=x.round(0), Naive=xn.round(0), kgCO2e_per_home=abate.round(0))).query("Homes > 0 or Naive > 0").sort_values("Homes", ascending=False)
        st.dataframe(plan.head(15), hide_index=True, use_container_width=True)
        fr = [(r, (solve(r) or {}).get("obj", np.nan) / 1000) for r in np.linspace(-20, 60, 9)]
        st.plotly_chart(px.line(pd.DataFrame(fr, columns=["Guaranteed worst-case saving $/yr per $1000", "tCO2e avoided/yr"]),
                                x="Guaranteed worst-case saving $/yr per $1000", y="tCO2e avoided/yr", markers=True,
                                title="Efficient frontier: how much abatement does each risk guarantee cost?"), use_container_width=True)
        st.caption("Scenarios share random prices/COP draws across places, so correlated risk (a gas-price crash hits every place) is respected.")


# ============================================================ 2. RESILIENT CITIES & BUILDINGS
@st.cache_data(show_spinner="Calibrating conformal intervals...")
def conformal_q(d, feats, alpha):
    mk = lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=.05, subsample=.8, random_state=0)
    return {t: mc.group_conformal(mk, d[feats].values, d[t].values, d["compact"].round(2).astype(str).values, alpha)[0] for t in ("HL", "CL")}


@st.cache_data(show_spinner="Counting heatwave events...")
def events(w):
    rows = []
    for loc, g in w.sort_values("Date").groupby("Location"):
        yrs = g.Tmean.notna().sum() / 365.25
        if yrs < 3:
            continue
        hw, sv = g.HW.values.astype(bool), g.SEV.values.astype(bool)
        starts = lambda b: int((b & ~np.r_[False, b[:-1]]).sum())
        rows.append(dict(Location=loc, years=yrs, events=starts(hw), severe=starts(sv)))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Reading daily maximum temperatures...")
def load_tmax(p):
    w = pd.read_csv(p, usecols=["Date", "Location", "MaxTemp"], parse_dates=["Date"])
    return w.dropna()


@st.cache_data(show_spinner="Reading Berkeley Earth (filtering to Australia/Pacific)...")
def load_berkeley(p):
    cols = pd.read_csv(p, nrows=0).columns
    pick = {c: next((k for k in ("dt", "AverageTemperature", "City", "Country") if re.sub(r"[^a-z0-9]", "", c.lower()) == k.lower()), None) for c in cols}
    use = [c for c, k in pick.items() if k]
    out = []
    for ch in pd.read_csv(p, usecols=use, chunksize=500000):
        ch = ch.rename(columns={c: pick[c] for c in use})
        out.append(ch[ch.Country.isin(["Australia", "New Zealand", "Fiji", "Papua New Guinea", "Samoa", "Tonga", "Vanuatu", "Solomon Islands"])])
    d = pd.concat(out)
    d["dt"] = pd.to_datetime(d.dt)
    return d.dropna(subset=["AverageTemperature"])


def resilience_tab(tab, X):
    T, enb = X["T"], X["enb"]
    with tab:
        st.subheader("Resilient cities & buildings maths")
        st.markdown("**1. Conformal prediction intervals - method 1.** Guaranteed-coverage ranges for building shapes the model has never seen.")
        a = st.slider("Miss rate allowed (alpha)", 0.05, 0.30, 0.10, 0.05)
        q = conformal_q(enb, X["FEATS"], a)
        m = st.columns(3)
        m[0].metric("Heating load, this home", f"{X['hl']:.1f}", f"{1 - a:.0%} range {X['hl'] - q['HL']:.1f} to {X['hl'] + q['HL']:.1f}")
        m[1].metric("Cooling load, this home", f"{X['cl']:.1f}", f"{1 - a:.0%} range {X['cl'] - q['CL']:.1f} to {X['cl'] + q['CL']:.1f}")
        m[2].metric("Half-width (heating / cooling)", f"+/-{q['HL']:.1f} / {q['CL']:.1f}")
        st.caption("Residuals come from holding out WHOLE building shapes (6 shape-folds). ENB2012 has only 12 shapes, so the guarantee is approximate - the simulation in the Proofs tab shows 88.7% achieved for a nominal 90%.")
        st.markdown("---\n**2. Empirical-Bayes shrinkage - method 2.** Severe-heatwave events are rare, so raw station counts are noisy; pooling gives better rates and honest rank probabilities.")
        ev = events(X["wx"])
        kind = st.radio("Count", ["severe", "events"], format_func=lambda s: "Severe heatwave events" if s == "severe" else "All heatwave events", horizontal=True)
        o = mc.eb_gamma_poisson(ev[kind].values, ev.years.values)
        rk = mc.posterior_ranks(o["pa"], o["pb"])
        ev = ev.assign(Raw=o["raw"], EB=o["mean"], Lo90=o["lo"], Hi90=o["hi"], P_top10=(rk < 10).mean(0), ExpRank=rk.mean(0) + 1)
        st.dataframe(ev.sort_values("EB", ascending=False)[["Location", "years", kind, "Raw", "EB", "Lo90", "Hi90", "P_top10", "ExpRank"]].head(15).round(2), hide_index=True, use_container_width=True)
        st.plotly_chart(px.scatter(ev, x="Raw", y="EB", hover_name="Location", title=f"Shrinkage toward the pooled rate (prior Gamma a={o['a']:.1f}, b={o['b']:.1f})"), use_container_width=True)
        st.markdown("---\n**3. SMAA - method 3.** How often is each place top-ranked if the judges disagree about the weights?")
        crit = T.set_index("Location")[["saving", "HW", "co2_35", "psave"]]
        Mx = ((crit - crit.min()) / (crit.max() - crit.min() + 1e-12)).values
        R = mc.smaa(Mx, 40000)
        sm = pd.DataFrame(dict(Place=crit.index, P_rank1=R[:, 0], P_top5=R[:, :5].sum(1), ExpectedRank=(R * np.arange(1, len(R) + 1)).sum(1)))
        sm["Current_triage_rank"] = T.set_index("Location").Triage.rank(ascending=False).reindex(sm.Place).values
        st.dataframe(sm.sort_values("P_top5", ascending=False).head(12).round(2), hide_index=True, use_container_width=True)
        st.caption("Criteria: yearly saving, heatwave days, CO2e avoided 2035, probability of positive saving. Weights ~ Dirichlet(1,1,1,1): every weighting equally likely. "
                   "Places with P(top-5) near 100% are robust choices; the sidebar slider is just one point in this space.")
        st.markdown("---\n**4. Heat extremes (GPD) and warming trend (Mann-Kendall, autocorrelation-corrected) - method 5.**")
        c1, c2 = st.columns(2)
        tx = load_tmax(X["wx_path"])
        loc = c1.selectbox("Station", sorted(tx.Location.unique()), index=sorted(tx.Location.unique()).index(X["city"]) if X["city"] in set(tx.Location) else 0)
        g = tx[tx.Location == loc].sort_values("Date").set_index("Date").MaxTemp.asfreq("D")
        yrs, u = g.notna().sum() / 365.25, g.quantile(0.95)
        exc = mc.decluster(g.fillna(-99).values, u)
        if len(exc) >= 15:
            ob = {Tr: mc.rl_bootstrap(exc - u, 0.0, yrs, Tr) for Tr in (10, 50)}
            c1.metric(f"1-in-10-year max temperature, {loc}", f"{u + ob[10]['rl']:.1f} C", f"90% CI {u + ob[10]['lo']:.1f} to {u + ob[10]['hi']:.1f}")
            c1.metric("1-in-50-year", f"{u + ob[50]['rl']:.1f} C", f"90% CI {u + ob[50]['lo']:.1f} to {u + ob[50]['hi']:.1f}")
            c1.caption(f"{len(exc)} declustered events above {u:.1f} C in {yrs:.1f} years; GPD shape {ob[10]['xi']:.2f}. Short records give wide intervals - that is the honest answer.")
        bp = ad.by_col("averagetemperature", "city", "country")
        if bp:
            b = load_berkeley(bp)
            city = c2.selectbox("City (Berkeley Earth)", sorted(b.City.unique()), index=0)
            gg = b[b.City == city].assign(y=lambda d: d.dt.dt.year).groupby("y").AverageTemperature.agg(["mean", "count"]).query("count >= 11")["mean"]
            gg = gg[gg.index >= 1950]
            if len(gg) > 20:
                sl = stats.theilslopes(gg.values, gg.index.values)
                nv, cr = mc.mk_test(gg.values, False), mc.mk_test(gg.values, True)
                c2.metric(f"Warming trend, {city} (since 1950)", f"{sl[0] * 10:+.2f} C per decade", f"Theil-Sen 95% CI {sl[2] * 10:+.2f} to {sl[3] * 10:+.2f}")
                c2.write(f"Mann-Kendall p-value: naive {nv['p']:.2g} -> autocorrelation-corrected {cr['p']:.2g} (variance inflation x{cr['infl']:.1f})")
                c2.plotly_chart(px.line(gg.reset_index(), x="y", y="mean"), use_container_width=True)
        else:
            with c2:
                ad.missing("Climate Change: Earth Surface Temperature Data (Berkeley Earth, GlobalLandTemperaturesByCity.csv)", "berkeleyearth/climate-change-earth-surface-temperature-data")


# ============================================================ 3. ZERO WASTE & GREEN INDUSTRIALISATION
@st.cache_data
def waste_comp(p):
    d = ad.read(p)
    cc = [c for c in d.columns if c.startswith("composition_") and c.endswith("percent")]
    return d, cc


def waste_industry_tab(tab, X):
    with tab:
        st.subheader("Zero waste & methane: which uncertainty matters most?  (methods 7)")
        wp = ad.by_col("compositionfood")
        lo_f, hi_f = 0.25, 0.55
        if wp:
            d, cc = waste_comp(wp)
            ic, fc = col(d, "income_id") or col(d, "income"), col(d, "composition_food")
            grp = st.selectbox("Income group (World Bank 'What a Waste')", sorted(d[ic].dropna().unique()))
            P = d[d[ic] == grp][cc].dropna(thresh=len(cc) - 2).fillna(0).values
            P = P[P.sum(1) > 50]
            al = mc.dirichlet_mle(P)
            j = cc.index(fc)
            b = stats.beta(al[j], al.sum() - al[j])
            lo_f, hi_f = float(b.ppf(.05)), float(b.ppf(.95))
            st.write(f"{len(P)} countries; Dirichlet MLE fitted. Food/organic share: mean {b.mean():.0%}, 90% interval {lo_f:.0%} to {hi_f:.0%}. "
                     f"Precision (sum of alphas) {al.sum():.1f}.")
            st.plotly_chart(px.bar(pd.DataFrame(dict(stream=[c.replace("composition_", "").replace("_percent", "") for c in cc], share=al / al.sum())),
                                   x="stream", y="share", title="Expected waste composition (Dirichlet mean)"), use_container_width=True)
            lm, nm = col(d, "total_msw_total_msw_generated"), col(d, "population_population")
            if lm and nm:
                dd = d[d[ic] == grp][[lm, nm]].apply(pd.to_numeric, errors="coerce").dropna()
                st.caption(f"Median generation in this group: {(dd[lm] / dd[nm] * 1000).median():,.0f} kg per person per year.")
        else:
            ad.missing("What a Waste Global Dataset", "mannmann2/what-a-waste-global-dataset")
        names = ["waste growth after 2026", "decay-rate multiplier k", "DOC multiplier", "DOC fraction decomposed", "methane correction factor", "gas capture by 2035", "food+garden diversion", "food share of waste"]
        bnd = [(0.0, 0.05), (0.5, 1.5), (0.7, 1.3), (0.4, 0.6), (0.6, 1.0), (0.0, 0.9), (0.0, 1.0), (lo_f, hi_f)]
        ton = st.number_input("Council landfilled tonnes per year", 5000, 2000000, 60000, 5000, key="ton2")
        f = lambda P_: mc.fod_vec(ton, 0.03, P_[:, 6], P_[:, 5], 28, growth_after=P_[:, 0], k_mult=P_[:, 1], doc_mult=P_[:, 2], docf=P_[:, 3], mcf=P_[:, 4], food_share=P_[:, 7])[:, -1]
        S1, ST = mc.sobol_indices(f, bnd, n=4096)
        sob = pd.DataFrame(dict(parameter=names, First_order=S1, Total=ST)).sort_values("Total")
        st.plotly_chart(px.bar(sob.melt("parameter"), y="parameter", x="value", color="variable", barmode="group", orientation="h",
                               title="Share of 2035 landfill-methane uncertainty explained by each input (Sobol indices)"), use_container_width=True)
        top = sob.iloc[-1]
        st.info(f"Biggest lever on 2035 methane uncertainty: **{top.parameter}** (total effect {top.Total:.0%}). Measure or manage that first - "
                f"a bin audit only helps if 'food share' is near the top. Method validated against the analytic Ishigami benchmark and an exact mass-balance check (Proofs tab).")
        st.markdown("---")
        green_tab(X)


def green_tab(X):
    st.subheader("Green industrialisation: how many times does a tonne get used?  (method 8)")
    k = st.columns(5)
    c, s, y = k[0].slider("Collected (c)", 0.1, 0.99, 0.80, 0.01), k[1].slider("Sorted & accepted (s)", 0.1, 0.99, 0.70, 0.01), k[2].slider("Reprocessing yield (y)", 0.1, 0.99, 0.90, 0.01)
    g, L = k[3].slider("Demand growth, %/yr", 0.0, 6.0, 2.0, 0.5) / 100, k[4].slider("Mean product life, years", 1, 40, 10)
    cs = mc.chain_stats(c, s, y)
    cmu = float(mc.cmu_gamma(cs["rho"], g, L, 2.0))
    need = 0.15 / float(mc.cmu_gamma(1.0, g, L, 2.0))
    r = np.random.default_rng(0)
    draw = lambda m: r.beta(m * 30, (1 - m) * 30, 5000)
    pc = float((mc.cmu_gamma(draw(c) * draw(s) * draw(y), g, L, 2.0) >= 0.15).mean())
    m = st.columns(5)
    m[0].metric("Expected uses per tonne", f"{cs['E_uses']:.2f}", "= 1/(1 - c*s*y)")
    m[1].metric("Virgin material per use", f"{cs['virgin_per_use']:.0%}")
    m[2].metric("Circular material use (CMU)", f"{cmu:.1%}", "goal 15% by 2035")
    m[3].metric("Loop closure needed for 15%", f"{need:.0%}", f"you have {cs['rho']:.0%}")
    m[4].metric("P(CMU >= 15%) with +/-10pt stage uncertainty", f"{pc:.0%}")
    stage = {"collection": c, "sorting": s, "reprocessing": y}
    weak = min(stage, key=stage.get)
    st.info(f"Weakest link: **{weak}** ({stage[weak]:.0%}). Because loop closure is a product c*s*y, +1 point on the weakest stage buys the most circularity per dollar. "
            f"Growth hurts: at {g:.0%} demand growth and {L}-year products, recycled supply lags demand, so CMU = closure x {mc.cmu_gamma(1.0, g, L, 2.0):.2f}.")
    if st.button("Verify live: simulate 100,000 tonnes through the loop"):
        st.write(f"Theory {cs['E_uses']:.3f} uses vs simulation {mc.chain_sim(c, s, y, 100000):.3f}")


# ============================================================ 4. AWARENESS: farming + message targeting
def farming_awareness_tab(tab, X):
    with tab:
        st.subheader("Awareness for farmers: how much does warming cost each crop?  (method 6, fixed-effects panel)")
        fp = ad.by_col("hghayield", "avgtemp", "pesticides")
        if fp:
            d = ad.read(fp).dropna()
            rn = col(d, "rain")
            crop = st.selectbox("Crop", sorted(d.item.unique()), index=0)
            dd = d[d.item == crop].copy()
            dd["ly"] = np.log(dd["hg_ha_yield"].clip(lower=1))
            Xm = np.column_stack([dd.avg_temp, dd.avg_temp ** 2, dd[rn] / 1000, (dd[rn] / 1000) ** 2, np.log1p(dd.pesticides_tonnes), dd.year - 2000])
            o = mc.fe_cluster(dd["ly"].values, Xm, dd.area.values)
            tcrit = stats.t.ppf(.975, o["G"] - 1)
            areas = sorted(dd.area.unique())
            area = st.selectbox("Place", areas, index=areas.index("Australia") if "Australia" in areas else 0)
            dT = st.slider("Warming, degC", 0.5, 4.0, 1.5, 0.5)
            T0 = dd[dd.area == area].avg_temp.mean()
            w = np.zeros(Xm.shape[1]); w[0], w[1] = dT, (T0 + dT) ** 2 - T0 ** 2
            eff, se = float(w @ o["b"]), float(np.sqrt(w @ o["V"] @ w))
            ols = np.linalg.lstsq(np.column_stack([np.ones(len(dd)), Xm]), dd["ly"].values, rcond=None)[0]
            eo = float(w @ ols[1:])
            m = st.columns(3)
            m[0].metric(f"{crop} yield change in {area} at +{dT:.1f} C", f"{np.expm1(eff):+.1%}", f"95% CI {np.expm1(eff - tcrit * se):+.1%} to {np.expm1(eff + tcrit * se):+.1%}")
            m[1].metric("Naive pooled regression says", f"{np.expm1(eo):+.1%}", "ignores place effects - biased (see Proofs #6)")
            m[2].metric("Temperature of peak yield", f"{-o['b'][0] / (2 * o['b'][1]):.1f} C" if o["b"][1] < 0 else "no interior peak")
            st.caption(f"{o['N']:,} observations, {o['G']} places, place fixed effects, cluster-robust SEs (t with {o['G'] - 1} df). Association, not controlled-experiment causation; "
                       f"annual national averages hide within-country variation.")
        else:
            ad.missing("Crop Yield Prediction Dataset (yield_df.csv)", "patelris/crop-yield-prediction-dataset")
        st.markdown("---\n**Climate education that reaches *everyone*: who is still persuadable, and which message works?** (method 10, Thompson sampling)")
        tp = ad.by_col("sentiment", "message")
        c1, c2 = st.columns(2)
        with c1:
            if tp:
                tw = pd.read_csv(tp, usecols=lambda c: c.lower() in ("sentiment", "message"))
                tw.columns = [c.lower() for c in tw.columns]
                lab = {2: "News", 1: "Pro-climate action", 0: "Neutral", -1: "Anti / sceptic"}
                cnt = tw.sentiment.value_counts().reindex([2, 1, 0, -1]).fillna(0)
                post = cnt.values + 1.0
                lo, hi = stats.beta.ppf(.05, post, post.sum() - post), stats.beta.ppf(.95, post, post.sum() - post)
                tab_ = pd.DataFrame(dict(Audience=[lab[i] for i in cnt.index], Messages=cnt.values, Share=post / post.sum(), Lo90=lo, Hi90=hi))
                st.dataframe(tab_.round(3), hide_index=True, use_container_width=True)
                pers = 1 - post[1] / post.sum() - post[0] / post.sum()
                st.metric("Share of climate conversation NOT already pro-action (Dirichlet posterior mean)", f"{pers:.0%}")
                st.caption("Tweets are a skewed sample of an online audience, not the population - use to size the awareness gap, then measure your own pilot.")
            else:
                ad.missing("Twitter Climate Change Sentiment Dataset", "edqian/twitter-climate-change-sentiment-dataset")
        with c2:
            st.markdown("Pilot design: assign each household one of 4 action-card framings; Thompson sampling shifts traffic to what works.")
            ps = [st.slider(f"Assumed true response rate, {n}", 0.05, 0.9, v, 0.01) for n, v in [("savings-first", .30), ("health & heat-safety", .34), ("community / neighbours", .38), ("plain-language checklist", .45)]]
            Tn = st.slider("Households in pilot", 200, 5000, 1000, 100)
            res = {k: mc.bandit_regret(ps, Tn, 150, k, seed=1) for k in ("ts", "uniform", "eps")}
            nm = {"ts": "Thompson sampling", "uniform": "Equal-split A/B", "eps": "Epsilon-greedy"}
            fig = go.Figure([go.Scatter(y=res[k][0], name=nm[k]) for k in res])
            fig.update_layout(xaxis_title="Households served", yaxis_title="Cumulative missed responses (regret)", height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Thompson sampling loses **{res['ts'][0][-1]:.0f}** responses vs **{res['uniform'][0][-1]:.0f}** for equal-split A/B "
                     f"({1 - res['ts'][0][-1] / res['uniform'][0][-1]:.0%} fewer) and gives the best card to {res['ts'][1][int(np.argmax(ps))]:.0%} of households.")


# ============================================================ 5. PROOFS + JUDGES' GUIDE
@st.cache_data(show_spinner="Running 10 simulation proofs (about 30 seconds)...")
def proofs_table():
    return proofs.run_all()


def proofs_tab(tab):
    with tab:
        st.subheader("Proofs: every method is checked against a known truth")
        st.write("Each test simulates data where the right answer is KNOWN, then checks the method finds it. No Kaggle data involved. Also runnable as `python proofs.py`.")
        if st.button("Run all 10 proofs now"):
            df = proofs_table()
            st.metric("Proofs passed", f"{(df.Result == 'PASS').sum()} / {len(df)}")
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.download_button("Download proof results (CSV)", df.to_csv(index=False), file_name="proof_results.csv")
        st.markdown("""
**What the proofs showed while building (honest log):** the landfill model from the first version double-counted each year's deposit (mass not conserved) and applied *policy* growth to past years - both fixed and now covered by an exact mass-balance test.
Shrinkage helps a lot for rare events (-31% error) and only a little for common ones (-8%). The S-curve forecast is calibrated only inside its regime (share not saturated near 0 or 100%), so the app refuses saturated countries.
Cluster-robust SEs need t(G-1) critical values with only ~40 clusters. Pooled OLS on crop data is badly biased; fixed effects are not.""")


def guide_tab(tab):
    with tab:
        st.subheader("Judges' guide: rubric -> evidence")
        st.markdown("""
| Criterion (weight) | Where to look |
|---|---|
| **COP31 alignment (30%)** | Scorecard (tab 10) maps all 5 priorities to the global 2035 goals with live numbers: electrification (tabs 1, 5, 11), waste & methane (6, 13), resilient cities & buildings (2, 3, 8, 12), green industrialisation (7, 13), awareness & farming (9, 14) |
| **Build quality (30%)** | 10 methods each validated by simulation against a known truth (tab 15); 12 real datasets; honest backtests; uncertainty everywhere; limitations stated |
| **Creativity (20%)** | CVaR-safe retrofit budget; SMAA triage that survives any judge's weights; Thompson-sampled pilot design; mass-balanced methane model with Sobol 'what to measure first' |
| **Presentation clarity (20%)** | Plain-language action card with reading-ease score; one question per tab; every chart has a one-line takeaway |
""")
        st.markdown("**Datasets - new in this version**")
        st.dataframe(status(), hide_index=True, use_container_width=True)
        st.markdown("""
**3-minute demo path:** (1) tab 1 map -> (2) tab 2 action card -> (3) tab 12 SMAA: *'change the weights, the answer holds'* -> (4) tab 12 conformal: *'honest ranges on unseen homes'* ->
(5) tab 11 CVaR: *'a budget with a bill-shock guarantee'* -> (6) tab 15 proofs: *'we tested the maths against known answers'* -> (7) tab 4 pilot.

**Known limits (say them first):** ENB2012 loads are simulated; Kaggle tweets are not a population sample; FAO yields are national annual averages; S-curve share is an OWID proxy;
heat data end 2017 in `weatherAUS`; GPD intervals on ~10 years of data are wide. The honest limitation slide is a scoring asset, not a weakness.
""")


def render(tabs, X):
    electrification_tab(tabs[0], X)
    resilience_tab(tabs[1], X)
    waste_industry_tab(tabs[2], X)
    farming_awareness_tab(tabs[3], X)
    proofs_tab(tabs[4])
    guide_tab(tabs[5])
