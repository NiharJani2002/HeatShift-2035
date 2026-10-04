"""
HeatShift 2035 - which homes and places should switch from gas to heat pumps FIRST,
and who is most exposed to heatwaves? (Climate Hack-tion 2026, COP31)

Priorities: Electrification (main) + Resilient Cities & Buildings + Awareness (plain-language action cards)

Data (Kaggle; put CSVs in this folder or ./data - core two can also be fetched by kagglehub):
  CORE      jsphyg/weather-dataset-rattle-package -> weatherAUS.csv | elikplim/eergy-efficiency-dataset -> ENB2012_data.csv
  ADD-ONS   EV Population Data, Global Methane Emissions, Flood Prediction, Crop Recommendation   (see addons.py)
  ADVANCED  anshtanwar/global-data-on-sustainable-energy | pralabhpoudel/world-energy-consumption | mannmann2/what-a-waste-global-dataset
            berkeleyearth/climate-change-earth-surface-temperature-data | patelris/crop-yield-prediction-dataset
            edqian/twitter-climate-change-sentiment-dataset                                       (see advanced.py)
Maths: mathcore.py (10 methods) validated by proofs.py (`python proofs.py`).

Run:  pip install streamlit pandas numpy scikit-learn scipy plotly
      streamlit run app.py
"""
import glob, os, re
import numpy as np, pandas as pd, streamlit as st, plotly.express as px
import addons, advanced
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

st.set_page_config(page_title="HeatShift 2035", layout="wide")
FEATS = ["compact", "surface", "wall", "roof", "height", "orient", "glaz", "glaz_dist"]
DELTAS = np.linspace(-0.8, 0.8, 9)          # COP uncertainty grid (index 4 = 0)
SLOPE, GAS_EF = 0.10, 0.185                  # COP gain per degC; kgCO2e per kWh gas burned (VERIFY vs NGA factors)
COORDS = {  # approximate station locations (for the map only)
 "Adelaide": (-34.9, 138.6), "Albany": (-35.0, 117.9), "Albury": (-36.1, 146.9), "AliceSprings": (-23.7, 133.9),
 "BadgerysCreek": (-33.9, 150.7), "Ballarat": (-37.6, 143.9), "Bendigo": (-36.8, 144.3), "Brisbane": (-27.5, 153.0),
 "Cairns": (-16.9, 145.8), "Canberra": (-35.3, 149.1), "Cobar": (-31.5, 145.8), "CoffsHarbour": (-30.3, 153.1),
 "Dartmoor": (-37.9, 141.3), "Darwin": (-12.5, 130.8), "GoldCoast": (-28.0, 153.4), "Hobart": (-42.9, 147.3),
 "Katherine": (-14.5, 132.3), "Launceston": (-41.4, 147.1), "Melbourne": (-37.8, 145.0), "MelbourneAirport": (-37.7, 144.8),
 "Mildura": (-34.2, 142.2), "Moree": (-29.5, 149.8), "MountGambier": (-37.8, 140.8), "MountGinini": (-35.5, 148.8),
 "Newcastle": (-32.9, 151.8), "Nhil": (-36.3, 141.7), "NorahHead": (-33.3, 151.6), "NorfolkIsland": (-29.0, 167.9),
 "Nuriootpa": (-34.5, 139.0), "PearceRAAF": (-31.7, 116.0), "Penrith": (-33.8, 150.7), "Perth": (-31.9, 115.9),
 "PerthAirport": (-31.9, 116.0), "Portland": (-38.3, 141.6), "Richmond": (-33.6, 150.8), "Sale": (-38.1, 147.1),
 "SalmonGums": (-33.0, 121.6), "Sydney": (-33.9, 151.2), "SydneyAirport": (-33.9, 151.2), "Townsville": (-19.3, 146.8),
 "Tuggeranong": (-35.4, 149.1), "Uluru": (-25.3, 131.0), "WaggaWagga": (-35.1, 147.4), "Walpole": (-35.0, 116.7),
 "Watsonia": (-37.7, 145.1), "Williamtown": (-32.8, 151.8), "Witchcliffe": (-34.0, 115.1), "Wollongong": (-34.4, 150.9),
 "Woomera": (-31.2, 136.8)}


# ----------------------------------------------------------------------------- data
def find(pattern, extra=()):
    for root in (".", "data", os.path.expanduser("~/.cache/kagglehub"), *extra):
        hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
        if hits:
            return hits[0]


def get_file(pattern, slug):
    p = find(pattern)
    if p:
        return p
    try:
        import kagglehub
        return find(pattern, extra=(kagglehub.dataset_download(slug),))
    except Exception:
        return None


@st.cache_data
def load_enb(path):
    d = pd.read_csv(path).iloc[:, :10].dropna()
    d.columns = FEATS + ["HL", "CL"]
    return d.astype(float)


@st.cache_data(show_spinner="Reading weather and flagging heatwaves...")
def load_weather(path):
    w = pd.read_csv(path, usecols=["Date", "Location", "MinTemp", "MaxTemp"], parse_dates=["Date"])
    parts = []
    for loc, g in w.groupby("Location"):
        g = g.drop_duplicates("Date").set_index("Date").sort_index().asfreq("D")
        g[["MinTemp", "MaxTemp"]] = g[["MinTemp", "MaxTemp"]].interpolate(limit=3, limit_area="inside")
        t = (g.MinTemp + g.MaxTemp) / 2
        d3 = t.rolling(3).mean()
        # Excess Heat Factor (Nairn & Fawcett): significance x acclimatisation (local ~10-yr record, not 30-yr)
        ehf = np.maximum(1, d3 - t.shift(3).rolling(30).mean()) * (d3 - t.quantile(0.95))
        pos = ehf[ehf > 0]
        thr = pos.quantile(0.85) if len(pos) >= 5 else np.inf
        parts.append(pd.DataFrame({"Date": g.index, "Location": loc, "Tmean": t.values,
                                   "HW": (ehf > 0).values, "SEV": (ehf >= thr).values}))
    return pd.concat(parts, ignore_index=True)


@st.cache_resource(show_spinner="Training and testing the building model...")
def fit_models(d):
    X, grp = d[FEATS], d["compact"].round(2).astype(str)
    mk = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
    out = {"models": {}, "rows": [], "sigma": {}}
    for t in ["HL", "CL"]:
        y = d[t]
        out["rows"].append(dict(Target=t, Test="Baseline: always guess the average", R2=0.0,
                                MAE=mean_absolute_error(y, np.full(len(y), y.mean()))))
        for name, cv in [("Random 5-fold (easy, leaky)", KFold(5, shuffle=True, random_state=0)),
                         ("Unseen building SHAPES (hard, honest)", GroupKFold(6))]:
            p = cross_val_predict(mk(), X, y, cv=cv, groups=grp)
            out["rows"].append(dict(Target=t, Test=name, R2=r2_score(y, p), MAE=mean_absolute_error(y, p)))
            if "SHAPES" in name:
                out["sigma"][t] = float(np.std(y - p))
        out["models"][t] = mk().fit(X, y)
    return out


@st.cache_data
def climate_table(w, cop7):
    rows, scops = [], {}
    for loc, g in w.groupby("Location"):
        g = g.dropna(subset=["Tmean"])
        yrs = len(g) / 365.25
        if yrs < 3:
            continue
        t = g.Tmean.to_numpy()
        hdd = np.maximum(0, 18 - t)
        tot = hdd.sum()
        scops[loc] = np.array([tot / (hdd / np.clip(cop7 + dl + SLOPE * (t - 7), 1.8, 5)).sum() if tot > 0 else cop7
                               for dl in DELTAS])
        rows.append(dict(Location=loc, HDD=tot / yrs, Tmean=t.mean(), HW=g.HW.sum() / yrs, SEV=g.SEV.sum() / yrs))
    return pd.DataFrame(rows), scops


# ----------------------------------------------------------------------------- load
enb_p, wx_p = get_file("ENB2012*.csv", "elikplim/eergy-efficiency-dataset"), get_file("weatherAUS*.csv", "jsphyg/weather-dataset-rattle-package")
if not (enb_p and wx_p):
    st.error("Dataset files not found. Download from Kaggle, unzip into this folder (or ./data):\n\n"
             "- weatherAUS.csv  <- kaggle.com/datasets/jsphyg/weather-dataset-rattle-package\n"
             "- ENB2012_data.csv <- kaggle.com/datasets/elikplim/eergy-efficiency-dataset")
    st.stop()
enb, wx = load_enb(enb_p), load_weather(wx_p)
M = fit_models(enb)
HL_REF, CL_MED = enb.HL.mean(), enb.CL.median()

# ----------------------------------------------------------------------------- assumptions
sb = st.sidebar
sb.header("Assumptions (edit them - judges will)")
elec = sb.slider("Electricity price, $/kWh", 0.15, 0.60, 0.30, 0.01)
gas = sb.slider("Gas price, $ per kWh of gas", 0.05, 0.30, 0.13, 0.01)
eff = sb.slider("Gas heater efficiency", 0.60, 0.95, 0.80, 0.01)
cop7 = sb.slider("Heat pump COP at 7 degC", 2.5, 5.0, 3.5, 0.1)
capex = sb.slider("Heat pump installed cost, $", 2000, 12000, 3500, 100)
area = sb.slider("Heated floor area, m2", 40, 250, 110, 5)
ef_now = sb.slider("Grid emissions today, kgCO2e/kWh", 0.10, 1.00, 0.55, 0.01)
ef_35 = sb.slider("Grid emissions 2035, kgCO2e/kWh", 0.00, 0.60, 0.15, 0.01)
w_heat = sb.slider("Triage weight: heating savings (1) vs heat risk (0)", 0.0, 1.0, 0.5, 0.05)
clim, scops = climate_table(wx, cop7)
locs = sorted(clim.Location)
anchor = sb.selectbox("Anchor climate for building loads", locs, index=locs.index("Melbourne") if "Melbourne" in locs else 0)
ref_hdd = max(float(clim.set_index("Location").loc[anchor, "HDD"]), 1.0)
sb.caption("ENB2012 loads are treated as annual loads in the anchor climate and scaled to other places by "
           "heating-degree-days. Absolute $ are indicative; the RANKING of places does not depend on this choice.")


def econ(hl, hdd, scop):
    dem = hl * area * hdd / ref_hdd
    hp = dem / scop
    return dict(demand=dem, saving=dem / eff * gas - hp * elec,
                co2_now=dem / eff * GAS_EF - hp * ef_now, co2_35=dem / eff * GAS_EF - hp * ef_35)


def mc(hl, hdd, sc_arr, n=3000):
    r = np.random.default_rng(7)
    e, g = elec * r.lognormal(0, .2, n), gas * r.lognormal(0, .25, n)
    ef, lm = r.uniform(.7, .9, n), r.lognormal(0, .3, n)
    sc = np.interp(r.normal(0, .3, n), DELTAS, sc_arr)
    dem = hl * area * hdd / ref_hdd * lm
    sav = dem / ef * g - dem / sc * e
    p10, p50, p90 = np.percentile(sav, [10, 50, 90])
    return p10, p50, p90, float((sav > 0).mean())


T = pd.concat([clim, pd.DataFrame([econ(HL_REF, r.HDD, scops[r.Location][4]) for r in clim.itertuples()])], axis=1)
T[["p10", "p50", "p90", "psave"]] = [mc(HL_REF, r.HDD, scops[r.Location]) for r in T.itertuples()]
T["SCOP"] = [scops[l][4] for l in T.Location]
T["Triage"] = 100 * (w_heat * T.saving.rank(pct=True) + (1 - w_heat) * T.HW.rank(pct=True))
hp_, rp_ = T.saving.rank(pct=True), T.HW.rank(pct=True)
T["Move"] = np.where((hp_ >= .5) & (rp_ >= .5), "Reverse-cycle heat pump (heating + cooling)",
                     np.where(hp_ >= rp_, "Electrify heating first", "Heat-resilience first (cooling access, shading)"))
T["lat"], T["lon"] = [T.Location.map(lambda l: COORDS.get(l, (np.nan, np.nan))[i]) for i in (0, 1)]

st.title("HeatShift 2035")
st.markdown("**Who it is for:** councils, community-housing providers and households deciding *where and which homes to "
            "upgrade first*.  **COP31 priorities:** Electrification (35% by 2035) - Resilient Cities & Buildings - "
            "Awareness (plain-language action cards).")
tabs = st.tabs(["1. Where first?", "2. Which home & action card", "3. Does it work? (tests)", "4. Pilot & adoption",
                "5. EVs & whole home", "6. Waste & methane", "7. Circular factory", "8. Floods", "9. Farming", "10. Scorecard & data",
                "11. Electrification maths", "12. Heat & buildings maths", "13. Waste & industry maths", "14. Farming & awareness maths",
                "15. Proofs", "16. Judges' guide"])
tab1, tab2, tab3, tab4 = tabs[:4]

# ----------------------------------------------------------------------------- tab 1
with tab1:
    st.subheader("Triage of 49 places: heat-pump payoff vs heatwave exposure")
    mp = T.dropna(subset=["lat"])
    fig = px.scatter_geo(mp, lat="lat", lon="lon", color="Move", size="Triage", size_max=20, hover_name="Location",
                         hover_data={"Triage": ":.0f", "saving": ":.0f", "HW": ":.1f", "lat": False, "lon": False})
    fig.update_geos(lataxis_range=[-45, -9], lonaxis_range=[112, 169], showcountries=True)
    st.plotly_chart(fig, use_container_width=True)
    top = T.sort_values("Triage").tail(15)
    st.plotly_chart(px.bar(top, x="Triage", y="Location", orientation="h", color="Move"), use_container_width=True)
    show = T.sort_values("Triage", ascending=False)[["Location", "Move", "Triage", "HDD", "SCOP", "saving", "p10", "p90",
                                                      "psave", "co2_now", "co2_35", "HW", "SEV"]]
    show.columns = ["Place", "First move", "Triage (0-100)", "Heating deg-days/yr", "Seasonal COP", "Saving $/yr (typical)",
                    "P10", "P90", "P(saving>0)", "CO2e saved kg/yr now", "CO2e saved kg/yr 2035", "Heatwave days/yr", "Severe heatwave days/yr"]
    st.dataframe(show.round(2), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Pick a place and a home design")
    c1, c2 = st.columns(2)
    city = c1.selectbox("Place", sorted(T.Location), index=sorted(T.Location).index("Melbourne") if "Melbourne" in set(T.Location) else 0)
    sh = enb.groupby("compact")[["surface", "wall", "roof", "height"]].first().reset_index().sort_values("compact", ascending=False)
    lab = [f"Compactness {r.compact:.2f} | {r.height:.1f} m tall | roof {r.roof:.0f} m2" for r in sh.itertuples()]
    s = sh.iloc[lab.index(c1.selectbox("Home shape", lab))]
    orient = c2.selectbox("Orientation code", sorted(enb.orient.unique()))
    gl = c2.selectbox("Glazing area (share of floor)", sorted(enb.glaz.unique()), index=2)
    gd = c2.selectbox("Glazing distribution code", sorted(enb.glaz_dist.unique()))
    row = dict(compact=s.compact, surface=s.surface, wall=s.wall, roof=s.roof, height=s.height, orient=orient, glaz=gl, glaz_dist=gd)
    hl, cl = (float(M["models"][t].predict(pd.DataFrame([row])[FEATS])[0]) for t in ("HL", "CL"))
    c = T.set_index("Location").loc[city]
    e = econ(hl, c.HDD, c.SCOP)
    p10, p50, p90, ps = mc(hl, c.HDD, scops[city])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Heating load (model)", f"{hl:.1f} +/- {M['sigma']['HL']:.1f}")
    m2.metric("Cooling load (model)", f"{cl:.1f} +/- {M['sigma']['CL']:.1f}")
    m3.metric("Saving, gas -> heat pump", f"${e['saving']:,.0f}/yr", f"P10-P90: ${p10:,.0f} to ${p90:,.0f}")
    m4.metric("Simple payback", f"{capex / p50:.1f} yrs" if p50 > 0 else "no payback", f"P(saving>0) = {ps:.0%}")
    gl2 = st.select_slider("What-if (PROXY for shading / better glass): glazing area becomes", sorted(enb.glaz.unique()), value=gl)
    hl2, cl2 = (float(M["models"][t].predict(pd.DataFrame([{**row, "glaz": gl2}])[FEATS])[0]) for t in ("HL", "CL"))
    st.caption(f"What-if result: heating load {hl2 - hl:+.1f}, cooling load {cl2 - cl:+.1f} (simulated by the model; a proxy, not a product spec).")
    tip = ("Plan for cooling: a reverse-cycle heat pump also cools, and shading west-facing windows helps."
           if c.HW >= T.HW.median() else "Cooling needs are lower here, but make sure the home stays safe in a heatwave.")
    card = (f"Your first move in {city}: {c.Move.lower()}.\n\n"
            f"- Heating: a {area} m2 home like this needs about {e['demand']:,.0f} kWh of heat a year. A heat pump gives about "
            f"{c.SCOP:.1f} units of heat for every 1 unit of electricity. Switching from gas could save about ${p50:,.0f} a year "
            f"(likely range ${p10:,.0f} to ${p90:,.0f}).\n"
            f"- Pollution: that avoids about {e['co2_now']:,.0f} kg of CO2e a year today, and about {e['co2_35']:,.0f} kg a year by 2035 as the grid gets cleaner.\n"
            f"- Hot days: {city} has about {c.HW:.0f} heatwave days a year. {tip} This home shape is "
            f"{'harder' if cl > CL_MED else 'easier'} than most to keep cool.\n"
            f"- Next step: ask an installer for a reverse-cycle quote, and check which rebates your state offers.")
    st.markdown("### Plain-language action card")
    st.info(card)
    words = re.findall(r"[A-Za-z']+", card)
    syl = sum(max(1, len(re.findall(r"[aeiouy]+", x.lower()))) for x in words)
    fre = 206.835 - 1.015 * len(words) / max(1, len(re.findall(r"[.!?]", card))) - 84.6 * syl / max(1, len(words))
    st.caption(f"Reading ease score: {fre:.0f} (60+ is plain English).")
    st.download_button("Download card", card, file_name=f"action_card_{city}.md")

# ----------------------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Test 1 - Does the building model work on shapes it has never seen?")
    mt = pd.DataFrame(M["rows"]).round(2)
    st.dataframe(mt, hide_index=True)
    st.caption("Random splits flatter the model (near-identical designs sit in both halves). Holding out whole building "
               "shapes is the honest test. Loads are simulated (ENB2012), not metered - see limitations.")
    st.subheader("Test 2 - Is the triage stable across time? (train 2007-12, test 2013-17)")
    rows = []
    for loc, g in wx.groupby("Location"):
        for p, gg in (("A", g[g.Date < "2013-01-01"]), ("B", g[g.Date >= "2013-01-01"])):
            gg = gg.dropna(subset=["Tmean"])
            if len(gg) / 365.25 < 2.5:
                continue
            t = gg.Tmean.to_numpy()
            hdd = np.maximum(0, 18 - t)
            sc = hdd.sum() / (hdd / np.clip(cop7 + SLOPE * (t - 7), 1.8, 5)).sum() if hdd.sum() > 0 else cop7
            rows.append(dict(Location=loc, P=p, saving=hdd.sum() / (len(gg) / 365.25) * (gas / eff - elec / sc), hw=gg.HW.sum() / (len(gg) / 365.25)))
    b = pd.DataFrame(rows).pivot(index="Location", columns="P", values=["saving", "hw"]).dropna()
    r_s, r_h = spearmanr(b[("saving", "A")], b[("saving", "B")])[0], spearmanr(b[("hw", "A")], b[("hw", "B")])[0]
    ov = len(set(b[("saving", "A")].nlargest(10).index) & set(b[("saving", "B")].nlargest(10).index))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rank agreement, heating payoff", f"{r_s:.2f}")
    k2.metric("Rank agreement, heatwave days", f"{r_h:.2f}")
    k3.metric("Top-10 places that stay top-10", f"{ov}/10")
    k4.metric("Places where saving>0 in 80%+ of draws", f"{(T.psave >= .8).mean():.0%}")
    st.plotly_chart(px.scatter(x=b[("saving", "A")], y=b[("saving", "B")], hover_name=b.index,
                               labels={"x": "Saving index 2007-12", "y": "Saving index 2013-17"}), use_container_width=True)
    st.caption("Heatwave days use a ~10-year local record (official BoM uses 30 years), so treat them as relative exposure.")
    st.subheader("Test 3 - Uncertainty is shown, not hidden")
    st.write("Every saving carries a P10-P90 range from 3,000 random draws over prices, heat pump performance, heater efficiency and load error.")

# ----------------------------------------------------------------------------- tab 4
with tab4:
    st.markdown("""
### Prototype -> real-world use
**Barrier:** councils and housing providers have no cheap way to decide *where* electrification pays off fastest and
who is most exposed to heat; households can't read tariff + climate + building data.
**Who uses it:** a council sustainability officer or community housing provider building a retrofit/electrification list.

**Pilot (8-12 weeks, ~30 homes in 2 contrasting climates):** (1) log current gas/electric smart-meter data; (2) run
HeatShift predictions *before* installs; (3) install heat pumps in half the homes first (stepped rollout);
(4) compare predicted vs metered heating energy (target: median error < 25%) and run the same triage rank
check on the pilot data; (5) survey whether the action card was understood (target: 80% can state their next step).

**What we'd measure:** prediction error vs meter, rank agreement of triage vs installer-reported priority,
bills saved, kgCO2e avoided, card comprehension.

**Limitations (be upfront):** ENB2012 loads are *simulated* for a single reference climate; weather data covers 2007-17
(no recent extremes); some places have no gas network; prices, COP curve and grid factors are editable assumptions.

**Disclosures:** Kaggle datasets - Rain in Australia (BoM), Energy Efficiency (UCI ENB2012, Tsanas & Xifara 2012), EV Population, Global Methane, Flood Prediction, Crop Recommendation, \nGlobal Data on Sustainable Energy, OWID World Energy Consumption, World Bank What a Waste, Berkeley Earth temperatures, FAO/World Bank crop yields, Twitter climate sentiment;\nlibraries - Streamlit, pandas, NumPy, scikit-learn, SciPy, Plotly; AI assistance - Claude (code drafting). Log all of this in your repo's DISCLOSURES file.
""")

# ----------------------------------------------------------------------------- all other priorities + advanced maths
X = dict(T=T, e=e, c=c, city=city, eff=eff, gas=gas, elec=elec, capex=capex, gas_ef=GAS_EF, ef_now=ef_now, ef_35=ef_35,
         wx=wx, wx_path=wx_p, scops=scops, DELTAS=DELTAS, enb=enb, FEATS=FEATS, hl=hl, cl=cl)
addons.render(tabs[4:10], X)
advanced.render(tabs[10:], X)
