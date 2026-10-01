"""Step 2: clean the 2024 reports and model what drives a building's energy use intensity (EUI).

    ./venv/bin/python analysis.py

Outcome: log of weather-normalised site EUI (kBtu per sq ft per year), so effects read as percentages.
Main model: multifamily housing (the majority of reporting buildings); a second model for offices to compare.
Diagnostics: variance inflation, Breusch-Pagan, residuals, Cook's distance, HC3 robust standard errors, a Huber
robust regression, and 10-fold cross-validated R-squared against a size-only baseline.
results/cleaning.json, results/models.json, results/coefficients.csv, results/diagnostics.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

HERE = Path(__file__).parent
ERAS = [(0, 1929, "pre-1930"), (1930, 1959, "1930-59"), (1960, 1979, "1960-79"), (1980, 1999, "1980-99"),
        (2000, 2009, "2000-09"), (2010, 2100, "2010+")]


def load():
    return pd.read_csv(HERE / "data" / "raw" / "ll84_2024.csv", na_values=["Not Available"], low_memory=False)


def clean(d, kind, eui_range):
    """Documented rules; each one's removals are counted."""
    steps = [("all 2024 reports", len(d))]
    rules = [
        ("existing buildings (not 'Test' or 'Design')", d["construction_status"] == "Existing"),
        ("standalone properties (campus members are reported again under their parent)",
         d["parent_property_id"].astype(str).str.startswith("Not Applicable")),
        (f"primary use: {kind}", d["primary_property_type"] == kind),
        ("weather-normalised EUI reported", d["weather_normalized_site_eui"].notna()),
        (f"EUI in a physically plausible range ({eui_range[0]}-{eui_range[1]} kBtu/sq ft)", d["weather_normalized_site_eui"].between(*eui_range)),
        ("floor area at least 25,000 sq ft (the law's threshold)", d["property_gfa_self_reported"] >= 25000),
        ("year built 1800-2024", d["year_built"].between(1800, 2024)),
        ("no default or estimated values, no data-quality alerts",
         (d["default_values"] != "Yes") & (d["estimated_data_flag"] != "Yes")
         & (d["alert_gross_floor_area_is"].fillna("Ok") == "Ok") & (d["alert_energy_meter_has_gaps"].fillna("Ok") == "Ok")),
        ("electricity share reported", d["percent_electricity"].notna()),
    ]
    keep = pd.Series(True, index=d.index)
    for name, cond in rules:
        keep &= cond.fillna(False)
        steps.append((name, int(keep.sum())))
    return d[keep].copy(), steps


def features(d, multifamily):
    f = pd.DataFrame(index=d.index)
    f["log_eui"] = np.log(d["weather_normalized_site_eui"])
    f["log_area"] = np.log(d["property_gfa_self_reported"])
    f["era"] = pd.cut(d["year_built"], [e[0] - 0.5 for e in ERAS] + [2100.5], labels=[e[2] for e in ERAS]).astype(str)
    f["electric_share"] = d["percent_electricity"] / 100
    f["steam"] = (d["district_steam_use_kbtu"].fillna(0) > 0).astype(int)
    oil = d[["fuel_oil_2_use_kbtu", "fuel_oil_4_use_kbtu", "fuel_oil_5_6_use_kbtu"]].fillna(0).sum(axis=1)
    f["oil"] = (oil > 0).astype(int)
    f["borough"] = d["borough"].fillna("Unknown").str.title()
    f["occupancy"] = d["occupancy"]
    if multifamily:
        f["units_per_10k_sqft"] = (d["multifamily_housing_total"] / d["property_gfa_self_reported"] * 10000).clip(upper=40)
    return f.dropna()


FORMULA = {
    True: "log_eui ~ log_area + C(era, Treatment('1960-79')) + electric_share + steam + oil + C(borough, Treatment('Manhattan')) + occupancy + units_per_10k_sqft",
    False: "log_eui ~ log_area + C(era, Treatment('1960-79')) + electric_share + steam + oil + C(borough, Treatment('Manhattan')) + occupancy",
}


def cv_r2(f, formula, k=10):
    pred = np.zeros(len(f))
    for tr, te in KFold(k, shuffle=True, random_state=0).split(f):
        pred[te] = smf.ols(formula, f.iloc[tr]).fit().predict(f.iloc[te])
    return 1 - ((f["log_eui"] - pred) ** 2).sum() / ((f["log_eui"] - f["log_eui"].mean()) ** 2).sum()


def fit(d, kind, eui_range):
    multifamily = kind == "Multifamily Housing"
    sub, steps = clean(d, kind, eui_range)
    f = features(sub, multifamily)
    formula = FORMULA[multifamily]
    ols = smf.ols(formula, f).fit()
    robust = ols.get_robustcov_results(cov_type="HC3")
    huber = smf.rlm(formula, f, M=sm.robust.norms.HuberT()).fit()
    X = ols.model.exog
    vif = {ols.model.exog_names[i]: float(variance_inflation_factor(X, i)) for i in range(1, X.shape[1])}
    bp = het_breuschpagan(ols.resid, X)
    infl = ols.get_influence().cooks_distance[0]
    coefs = pd.DataFrame({"term": ols.model.exog_names, "coef": ols.params.values, "se_hc3": robust.bse,
                          "p_hc3": robust.pvalues, "huber_coef": huber.params.values})
    coefs["pct_effect"] = 100 * (np.exp(coefs["coef"]) - 1)
    coefs.insert(0, "model", kind)
    diag = {"n": int(ols.nobs), "r2": ols.rsquared, "adj_r2": ols.rsquared_adj,
            "cv_r2": cv_r2(f, formula), "cv_r2_size_only": cv_r2(f, "log_eui ~ log_area"),
            "breusch_pagan_p": float(bp[1]), "max_vif": max(vif.values()), "vif": vif,
            "cooks_over_4_over_n": int((infl > 4 / len(f)).sum()), "max_cooks": float(infl.max()),
            "median_eui": float(np.exp(f["log_eui"]).median()),
            "largest_huber_ols_gap": float((coefs["huber_coef"] - coefs["coef"]).abs().max())}
    resid = pd.DataFrame({"fitted": ols.fittedvalues, "resid": ols.resid, "cooks": infl})
    return steps, coefs, diag, resid, f


def robustness(d):
    """Do the headline multifamily effects survive (a) keeping the estimated/default-value buildings and
    (b) switching to source EUI, which counts power-plant losses (grid electricity ~2.8x site energy)?"""
    base, _ = clean(d, "Multifamily Housing", (10, 500))
    loose = d[(d["construction_status"] == "Existing") & d["parent_property_id"].astype(str).str.startswith("Not Applicable")
              & (d["primary_property_type"] == "Multifamily Housing") & d["weather_normalized_site_eui"].between(10, 500)
              & (d["property_gfa_self_reported"] >= 25000) & d["year_built"].between(1800, 2024) & d["percent_electricity"].notna()]
    rows = []
    for label, sub, outcome in [("main: clean set, site EUI", base, "weather_normalized_site_eui"),
                                ("keep estimated/default-value buildings", loose, "weather_normalized_site_eui"),
                                ("clean set, source EUI", base, "source_eui_kbtu_ft")]:
        f = features(sub, True)
        f["log_eui"] = np.log(sub.loc[f.index, outcome])
        f = f[np.isfinite(f["log_eui"])]
        m = smf.ols(FORMULA[True], f).fit(cov_type="HC3")
        get = lambda term: 100 * (np.exp(m.params[term]) - 1)
        rows.append({"version": label, "n": int(m.nobs), "r2": round(m.rsquared, 3),
                     "electric_share_per_10pts": round(100 * (np.exp(m.params["electric_share"] * 0.1) - 1), 1),
                     "steam": round(get("steam"), 1), "built_2000_09": round(get("C(era, Treatment('1960-79'))[T.2000-09]"), 1),
                     "built_2010_plus": round(get("C(era, Treatment('1960-79'))[T.2010+]"), 1),
                     "built_pre_1930": round(get("C(era, Treatment('1960-79'))[T.pre-1930]"), 1)})
    return pd.DataFrame(rows)


def confounding():
    """Raw era gap vs the gap once fuel mix is held equal: newer buildings are far more electric."""
    f = pd.read_csv(HERE / "results" / "model_data_multifamily.csv", index_col=0)
    term = "C(era, Treatment('1960-79'))[T.2010+]"
    pct = lambda m: round(100 * (np.exp(m.params[term]) - 1), 1)
    return {"median_electric_share_by_era": f.groupby("era")["electric_share"].median().round(2).to_dict(),
            "built_2010_plus_era_only": pct(smf.ols("log_eui ~ C(era, Treatment('1960-79'))", f).fit()),
            "built_2010_plus_with_electric_share": pct(smf.ols("log_eui ~ C(era, Treatment('1960-79')) + electric_share", f).fit())}


def main():
    d = load()
    (HERE / "results").mkdir(exist_ok=True)
    out_steps, out_diag, all_coefs = {}, {}, []
    for kind, rng in [("Multifamily Housing", (10, 500)), ("Office", (10, 1000))]:
        steps, coefs, diag, resid, f = fit(d, kind, rng)
        out_steps[kind], out_diag[kind] = steps, diag
        all_coefs.append(coefs)
        slug = "multifamily" if kind.startswith("Multi") else "office"
        resid.to_csv(HERE / "results" / f"residuals_{slug}.csv")
        f.to_csv(HERE / "results" / f"model_data_{slug}.csv")
    pd.concat(all_coefs).round(5).to_csv(HERE / "results" / "coefficients.csv", index=False)
    (HERE / "results" / "confounding.json").write_text(json.dumps(confounding(), indent=2) + "\n")
    rob = robustness(d)
    rob.to_csv(HERE / "results" / "robustness.csv", index=False)
    print(rob.to_string(index=False))
    (HERE / "results" / "cleaning.json").write_text(json.dumps(out_steps, indent=2) + "\n")
    (HERE / "results" / "diagnostics.json").write_text(json.dumps(out_diag, indent=2, default=float) + "\n")
    for kind in out_steps:
        print(kind)
        for name, n in out_steps[kind]:
            print(f"   {n:7,}  {name}")
        dg = out_diag[kind]
        print(f"   R2 {dg['r2']:.3f}  CV R2 {dg['cv_r2']:.3f} (size only {dg['cv_r2_size_only']:.3f})  BP p {dg['breusch_pagan_p']:.2g}  max VIF {dg['max_vif']:.1f}  median EUI {dg['median_eui']:.1f}")
    print(pd.concat(all_coefs)[["model", "term", "pct_effect", "p_hc3", "huber_coef", "coef"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
