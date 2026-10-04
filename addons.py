"""
HeatShift add-ons: covers the other COP31 priorities. Every module works with whatever Kaggle CSVs you drop in
this folder or ./data (files are found by their COLUMNS, not file names) and degrades gracefully if one is missing.
"""
import os, re
import numpy as np, pandas as pd, streamlit as st, plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

SKIP = ("venv", ".venv", "site-packages", ".git", "node_modules", "__pycache__")
TAGS = {"Electrification": ["electric", "charging", "energy", "ev_"],
        "Zero Waste & Methane": ["waste", "methane", "landfill", "emission"],
        "Resilient Cities & Buildings": ["flood", "heat", "building", "weather", "temperature", "rain"],
        "Green Industrialization": ["recycl", "circular", "material", "steel", "plastic", "manufactur"],
        "Awareness / Farming": ["crop", "farm", "yield", "soil", "agri", "climate"]}


def csvs():
    out = []
    for dp, dn, fs in os.walk("."):
        dn[:] = [d for d in dn if d not in SKIP]
        out += [os.path.join(dp, f) for f in fs if f.lower().endswith(".csv")]
    return sorted(out)


def squash(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())


def by_col(*need):
    """First CSV whose headers contain every fragment in `need` (ignoring case/punctuation)."""
    for p in csvs():
        try:
            cols = [squash(c) for c in pd.read_csv(p, nrows=0).columns]
        except Exception:
            continue
        if all(any(n in c for c in cols) for n in need):
            return p


@st.cache_data
def read(p):
    d = pd.read_csv(p)
    d.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_") for c in d.columns]
    return d


def missing(what, slug_hint):
    st.info(f"Optional dataset not found: {what}. Download from Kaggle ({slug_hint}), unzip into this folder or ./data, rerun.")


# ============================================================ 5. Electrification: whole-home passport + EVs
def ev_tab(tab, X, S):
    with tab:
        st.subheader("Electrification beyond the heater: the whole-home passport")
        p = by_col("electricvehicletype", "modelyear")
        if p:
            d = read(p)
            d = d[d.model_year >= 2010].copy()
            d["bev"] = d.electric_vehicle_type.astype(str).str.contains("BEV")
            g = d.groupby("model_year").agg(registrations=("bev", "size"), bev_share=("bev", "mean")).reset_index()
            full = g[g.model_year < g.model_year.max()]
            if len(full) >= 6 and full.registrations.iloc[-6] > 0:
                gr = (full.registrations.iloc[-1] / full.registrations.iloc[-6]) ** (1 / 5) - 1
                st.metric("EV registrations growth (last 5 full model years)", f"{gr:.0%} per year")
            st.plotly_chart(px.bar(g, x="model_year", y="registrations", color="bev_share",
                                   title="EV registrations by model year (colour = battery-only share)"), use_container_width=True)
            st.caption("Washington State (USA) registry used as an adoption-speed signal for transport electrification - not an Australian forecast.")
        else:
            missing("Electric Vehicle Population Data", "search that title")
        c1, c2, c3 = st.columns(3)
        km, lper = c1.slider("Car km/year", 3000, 30000, 12000, 500), c1.slider("Petrol L/100km", 4.0, 14.0, 9.0, 0.5)
        kwh100, ppl = c2.slider("EV kWh/100km", 12.0, 25.0, 18.0, 0.5), c2.slider("Petrol $/L", 1.5, 3.0, 2.0, 0.05)
        base = c3.slider("Other electricity, kWh/yr", 1500, 8000, 4000, 100)
        hp_on, ev_on = c3.checkbox("Heat pump replaces gas", True), c3.checkbox("EV replaces petrol car", True)
        dem, gas_in, hp = X["e"]["demand"], X["e"]["demand"] / X["eff"], X["e"]["demand"] / X["c"].SCOP
        litres, ev = km * lper / 100, km * kwh100 / 100
        pet_kwh = litres * 9.5
        before = base + gas_in + pet_kwh
        el_after = base + (hp if hp_on else 0) + (ev if ev_on else 0)
        fuel_after = (0 if hp_on else gas_in) + (0 if ev_on else pet_kwh)
        sh0, sh1 = base / before, el_after / (el_after + fuel_after)

        def co2(ef):
            b = gas_in * X["gas_ef"] + litres * 2.3 + base * ef
            a = (0 if hp_on else gas_in) * X["gas_ef"] + (0 if ev_on else litres) * 2.3 + el_after * ef
            return b - a
        dollars = (gas_in * X["gas"] + litres * ppl + base * X["elec"]) - ((0 if hp_on else gas_in) * X["gas"] + (0 if ev_on else litres) * ppl + el_after * X["elec"])
        m = st.columns(4)
        m[0].metric("Household energy that is electric", f"{sh0:.0%} -> {sh1:.0%}", "global goal: 35% by 2035")
        m[1].metric("Final energy used", f"{1 - (el_after + fuel_after) / before:+.0%}")
        m[2].metric("CO2e avoided per year", f"{co2(X['ef_now']):,.0f} kg now", f"{co2(X['ef_35']):,.0f} kg by 2035")
        m[3].metric("Running-cost change", f"${dollars:,.0f}/yr")
        S["elec"], S["bldg_cut"] = (sh0, sh1), 1 - X["eff"] / X["c"].SCOP


# ============================================================ 6. Zero waste & methane (IPCC first-order decay)
COMP = {"Food": (0.40, 0.15, 0.185), "Garden": (0.15, 0.20, 0.10), "Paper": (0.15, 0.40, 0.06),
        "Wood": (0.05, 0.43, 0.03), "Textiles": (0.05, 0.24, 0.06)}  # share, DOC, k (IPCC 2006 temperate-wet defaults - VERIFY)


def fod(tonnes, growth, divert, capture, gwp, years=range(2011, 2036), growth_after=None):
    """IPCC first-order decay via mathcore.fod_vec (mass-conserving; history uses today's growth, policy growth only after 2026)."""
    import mathcore
    out = mathcore.fod_vec(tonnes, growth, divert, capture, gwp, growth_after=growth_after, years=years)[0]
    return pd.Series(out, index=list(years)).loc[2026:]


def waste_tab(tab, S):
    with tab:
        st.subheader("Zero waste & methane: what does a council's landfill do to the climate by 2035?")
        c = st.columns(5)
        tonnes = c[0].number_input("Landfilled tonnes/yr (council)", 5000, 2000000, 60000, 5000)
        g = c[1].slider("Waste growth if nothing changes, %/yr", 0.0, 5.0, 3.0, 0.1) / 100
        div = c[2].slider("Food+garden diverted by 2035, %", 0, 100, 70, 5) / 100
        cap = c[3].slider("Landfill gas captured by 2035, %", 0, 90, 60, 5) / 100
        gwp = c[4].selectbox("Methane GWP (x CO2)", [28, 81], help="28 = 100-yr, 81 = 20-yr (AR6 ~ 80)")
        bau = fod(tonnes, g, 0, 0, gwp)
        pol = fod(tonnes, g, div, cap, gwp, growth_after=g / 2)   # policy halves waste growth AFTER 2026 (the COP31 goal)
        df = pd.DataFrame({"Business as usual": bau, "With policy": pol}).reset_index().melt("index")
        st.plotly_chart(px.line(df, x="index", y="value", color="variable",
                                labels={"index": "Year", "value": "tCO2e from landfill methane"}), use_container_width=True)
        cut = 1 - pol.loc[2035] / bau.loc[2035]
        m = st.columns(3)
        m[0].metric("Landfill methane cut by 2035", f"{cut:.0%}")
        m[1].metric("Waste growth rate", f"{g:.1%} -> {g / 2:.1%}", "goal: slow growth by 50%")
        m[2].metric("CO2e avoided 2026-35", f"{(bau - pol).sum():,.0f} t")
        S["ch4"] = cut
        st.caption("IPCC first-order-decay model with default parameters (editable in code). Treat results as screening estimates; "
                   "replace the composition with your council's bin audit.")
        mp = next((x for x in csvs() if "methane" in os.path.basename(x).lower()), None)
        if mp:
            d = read(mp)
            txt = d.select_dtypes("object")
            mask = txt.apply(lambda s: s.str.contains("waste", case=False, na=False)).any(axis=1) if not txt.empty else pd.Series(False, index=d.index)
            st.markdown(f"**Global methane data ({os.path.basename(mp)})**: {int(mask.sum())} waste-related rows of {len(d)}.")
            st.dataframe((d[mask] if mask.any() else d).head(50), use_container_width=True)
        else:
            missing("Global Methane Emissions (1975-2022)", "search that title")


# ============================================================ 7. Green industrialisation (circular materials)
def circ_tab(tab, S):
    with tab:
        st.subheader("Green industrialisation: how circular is a factory's material use?")
        d0 = pd.DataFrame({"material": ["Steel", "Aluminium", "Copper", "PET plastic", "Glass", "Paper/board"],
                           "t_per_year": [10000, 2000, 500, 3000, 4000, 1500],
                           "rec_now": [15, 10, 20, 5, 25, 40], "rec_2035": [45, 55, 50, 35, 60, 75],
                           "virgin": [2300, 16000, 3500, 2200, 850, 1100], "recycled": [600, 900, 800, 900, 600, 800]})
        ed = st.data_editor(d0, num_rows="dynamic", use_container_width=True).dropna()
        t = ed.t_per_year.astype(float)
        if t.sum() > 0:
            cm = lambda r: (t * r / 100).sum() / t.sum()
            em = lambda r: (t * (r / 100 * ed.recycled + (1 - r / 100) * ed.virgin)).sum() / 1000
            c0, c1 = cm(ed.rec_now), cm(ed.rec_2035)
            m = st.columns(3)
            m[0].metric("Circular material use", f"{c0:.0%} -> {c1:.0%}", "global goal: 15% by 2035")
            m[1].metric("Embodied CO2e", f"{em(ed.rec_now):,.0f} -> {em(ed.rec_2035):,.0f} t")
            m[2].metric("CO2e avoided per year", f"{em(ed.rec_now) - em(ed.rec_2035):,.0f} t")
            S["cmu"] = (c0, c1)
        st.caption("Emission factors are indicative placeholders (kgCO2e per tonne) - replace with ICE / EPD values for your products. "
                   "The editable table is the point: a plant manager can enter their own tonnes and recycled-content targets.")


# ============================================================ 8. Resilient cities: flood levers
@st.cache_resource(show_spinner="Training flood-risk model...")
def flood_model(p):
    d = read(p).drop(columns=["id"], errors="ignore").select_dtypes("number").dropna()
    tgt = [c for c in d.columns if "floodprobability" in c.replace("_", "")][0]
    d = d.sample(min(len(d), 15000), random_state=0)
    X, y = d.drop(columns=tgt), d[tgt]
    m = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=0)
    r2 = cross_val_score(m, X, y, cv=KFold(3, shuffle=True, random_state=0), scoring="r2").mean()
    m.fit(X, y)
    return m, list(X.columns), float(r2), X.median(), X.min()


def flood_tab(tab, S):
    with tab:
        st.subheader("Resilient cities: which flood levers should a council pull first?")
        p = by_col("floodprobability")
        if not p:
            missing("Flood Prediction Dataset (has a FloodProbability column)", "search 'flood prediction'")
            return
        m, cols, r2, med, mn = flood_model(p)
        imp = pd.Series(m.feature_importances_, cols).sort_values(ascending=False)
        st.plotly_chart(px.bar(imp.head(8)[::-1], orientation="h", labels={"value": "importance", "index": "factor"}), use_container_width=True)
        st.caption(f"Held-out R2 = {r2:.2f} (3-fold). Factors are index scores in the dataset; this shows association, not proof of cause.")
        top = list(imp.index[:3])
        base = pd.DataFrame([med])[cols]
        scn = base.copy()
        sl = st.columns(3)
        for i, f in enumerate(top):
            cut = sl[i].slider(f"Improve '{f}' by (points)", 0, 5, 2)
            scn[f] = max(float(mn[f]), float(med[f]) - cut)
        b, s = float(m.predict(base)[0]), float(m.predict(scn)[0])
        st.metric("Predicted flood probability", f"{b:.2f} -> {s:.2f}", f"{(s - b) / b:+.0%}" if b else None)
        S["flood"] = (s - b) / b if b else 0


# ============================================================ 9. Awareness: climate-resilient farming
@st.cache_resource(show_spinner="Training crop model...")
def crop_model(p):
    d = read(p)
    X, y = d[["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]], d["label"]
    m = RandomForestClassifier(200, random_state=0, n_jobs=-1)
    acc = cross_val_score(m, X, y, cv=KFold(5, shuffle=True, random_state=0)).mean()
    return m.fit(X, y), float(acc), X.median(), list(X.columns)


@st.cache_data
def station_clim(wx_path):
    w = pd.read_csv(wx_path, usecols=["Location", "MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm"])
    g = w.groupby("Location").mean()
    return pd.DataFrame({"temperature": (g.MinTemp + g.MaxTemp) / 2, "rainfall": g.Rainfall * 30.4,
                         "humidity": (g.Humidity9am + g.Humidity3pm) / 2}).dropna()


def farm_tab(tab, X, S):
    with tab:
        st.subheader("Awareness for farmers: will today's best crop still suit this place in a hotter, drier climate?")
        p = by_col("label", "rainfall", "humidity", "ph")
        if not p:
            missing("Crop Recommendation Dataset (N, P, K, temperature, humidity, ph, rainfall, label)", "search 'crop recommendation'")
            return
        m, acc, med, cols = crop_model(p)
        cl = station_clim(X["wx_path"])
        c = st.columns(4)
        place = c[0].selectbox("Farm location (BoM station)", list(cl.index), index=list(cl.index).index(X["city"]) if X["city"] in cl.index else 0)
        warm = c[1].slider("Warming, degC", 0.0, 3.0, 1.5, 0.5)
        rain = c[2].slider("Rainfall change, %", -30, 20, -15, 5)
        ph = c[3].slider("Soil pH", 4.0, 9.0, float(med["ph"]), 0.1)
        r = cl.loc[place]
        row = dict(n=med["n"], p=med["p"], k=med["k"], temperature=r.temperature, humidity=r.humidity, ph=ph, rainfall=r.rainfall)
        now = pd.Series(m.predict_proba(pd.DataFrame([row])[cols])[0], m.classes_).sort_values(ascending=False)
        row2 = {**row, "temperature": r.temperature + warm, "rainfall": r.rainfall * (1 + rain / 100)}
        fut = pd.Series(m.predict_proba(pd.DataFrame([row2])[cols])[0], m.classes_).sort_values(ascending=False)
        a, b = st.columns(2)
        a.markdown("**Today - top 3 crops**"); a.dataframe(now.head(3).rename("suitability").to_frame().style.format("{:.0%}"))
        b.markdown("**Stress-tested - top 3 crops**"); b.dataframe(fut.head(3).rename("suitability").to_frame().style.format("{:.0%}"))
        flip = now.index[0] != fut.index[0]
        S["crop"] = flip
        st.warning(f"At +{warm:.1f} degC and {rain:+d}% rain, the best-fit crop changes from {now.index[0]} to {fut.index[0]}: trial diversification."
                   if flip else f"{now.index[0]} stays the best fit, but its suitability is {now.iloc[0]:.0%} -> {fut.get(now.index[0], 0):.0%}.")
        st.caption(f"Cross-validated accuracy {acc:.0%}. The dataset's crop list (rice, mango, coconut, banana, maize...) is not an Australian "
                   "broadacre list - treat as a demonstration of the method; swap in a local crop dataset to deploy.")


# ============================================================ 10. Quiz, scorecard, data library
def library():
    rows = []
    for p in csvs():
        try:
            d = pd.read_csv(p, nrows=2000)
        except Exception:
            continue
        key = (os.path.basename(p) + " " + " ".join(map(str, d.columns))).lower()
        rows.append(dict(file=p, columns=d.shape[1], priorities=", ".join(k for k, v in TAGS.items() if any(w in key for w in v)) or "-"))
    return pd.DataFrame(rows)


def score_tab(tab, X, S):
    with tab:
        st.subheader("Teach-back quiz (climate knowledge for everyone)")
        T, c = X["T"], X["c"]
        q1 = list(T.sample(4, random_state=1).Location)
        a1 = T.set_index("Location").loc[q1].HW.idxmax()
        for key, q, opts, ans in [("q1", "Which of these places has the MOST heatwave days a year?", q1, a1),
                                  ("q2", "As the grid gets cleaner, a heat pump's yearly CO2 saving will be...", ["Bigger", "Smaller"],
                                   "Bigger" if X["e"]["co2_35"] > X["e"]["co2_now"] else "Smaller"),
                                  ("q3", f"What is the suggested first move in {X['city']}?",
                                   sorted(T.Move.unique()), c.Move)]:
            pick = st.radio(q, opts, index=None, key=key)
            if pick:
                (st.success if pick == ans else st.error)("Correct!" if pick == ans else f"Not quite - it is {ans}.")
        st.subheader("COP31 scorecard (live numbers from this app)")
        f = lambda v, fmt: fmt.format(*v) if v else "open that tab"
        rows = [("Electrification", "35% by 2035", f(S.get("elec"), "{:.0%} -> {:.0%} of household energy electric")),
                ("Zero Waste & Methane", "slow waste growth 50%", f"landfill methane cut {S['ch4']:.0%}" if "ch4" in S else "open tab 6"),
                ("Resilient Cities & Buildings", "-25% building energy", f"heating energy {-S['bldg_cut']:+.0%}; flood risk {S['flood']:+.0%}" if "flood" in S else
                 f"heating energy {-S.get('bldg_cut', 0):+.0%}"),
                ("Green Industrialization", "15% circular use", f(S.get("cmu"), "{:.0%} -> {:.0%} circular material use")),
                ("Awareness Across All Areas", "resilient farming + education",
                 ("crop choice flips under stress" if S.get("crop") else "crop choice holds under stress") if "crop" in S else "open tab 9")]
        st.table(pd.DataFrame(rows, columns=["Priority", "Global goal", "What this prototype shows"]))
        st.subheader("Data library and auto-disclosure")
        lib = library()
        st.dataframe(lib, use_container_width=True, hide_index=True)
        md = ["# DISCLOSURES", "", "## Datasets (Kaggle)"] + [f"- `{r.file}` - priorities: {r.priorities}" for r in lib.itertuples()] + \
             ["", "## Libraries", "Streamlit, pandas, NumPy, scikit-learn, SciPy, Plotly",
              "", "## AI tools", "Claude (Anthropic) - drafted code; team reviewed and tested it.",
              "", "## Assumptions", "Prices, COP curve, grid factors, IPCC FOD defaults and emission factors are editable assumptions."]
        st.download_button("Download DISCLOSURES.md", "\n".join(md), file_name="DISCLOSURES.md")


def render(tabs, X):
    S = {}
    ev_tab(tabs[0], X, S)
    waste_tab(tabs[1], S)
    circ_tab(tabs[2], S)
    flood_tab(tabs[3], S)
    farm_tab(tabs[4], X, S)
    score_tab(tabs[5], X, S)
