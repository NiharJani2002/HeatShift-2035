# HeatShift 2035

I built HeatShift 2035 for Climate Hack-tion 2026 (COP31). It answers one practical question: which homes and places should switch from gas heating to heat pumps first, and who is most exposed to heatwaves?

I used Kaggle's Australian weather data to work out heating degree-days, heat pump performance and heatwave days for 49 places. A gradient boosting model trained on the Energy Efficiency (ENB2012) dataset predicts the heating and cooling load of a home design. Together they give a yearly saving, a CO2 figure and a plain-language action card. Prices, COP and grid emissions are sliders, so anyone can change my assumptions.

There are extra tabs for EVs, landfill methane, circular industry, floods and farming. Ten methods sit in `mathcore.py`: conformal intervals, empirical-Bayes shrinkage, SMAA, a CVaR linear programme, extreme values, fixed-effects panels, Sobol sensitivity, Markov chains, an S-curve forecast and Thompson sampling.

I test every method on simulated data where the true answer is known. Run `python proofs.py`. All 10 pass.

**Run it**

```
pip install streamlit pandas numpy scikit-learn scipy plotly
streamlit run app.py
```

Put the Kaggle CSVs in this folder or `./data`.

**Limits:** ENB2012 loads are simulated, not metered, and the weather only covers 2007-17.

*Nihar Mahesh Jani*
