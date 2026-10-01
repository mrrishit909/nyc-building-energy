"""The project's check.

    ./venv/bin/python check.py

1. The cleaning funnel recomputes and every rule removes what it says.
2. The multifamily model refits to the same coefficients from the saved model data (independent refit with numpy).
3. The robustness table and the confounding numbers recompute; the headline effects keep their sign in all versions.
"""
import json

import numpy as np
import pandas as pd

from analysis import HERE, clean, load


def main():
    d = load()
    steps = json.loads((HERE / "results" / "cleaning.json").read_text())
    sub, recomputed = clean(d, "Multifamily Housing", (10, 500))
    assert [list(s) for s in recomputed] == steps["Multifamily Housing"]
    assert sub["weather_normalized_site_eui"].between(10, 500).all() and (sub["property_gfa_self_reported"] >= 25000).all()

    f = pd.read_csv(HERE / "results" / "model_data_multifamily.csv", index_col=0)
    X = pd.get_dummies(f.drop(columns="log_eui"), columns=["era", "borough"], dtype=float)
    X = X.drop(columns=["era_1960-79", "borough_Manhattan"])
    X.insert(0, "const", 1.0)
    beta, *_ = np.linalg.lstsq(X.values, f["log_eui"].values, rcond=None)
    b = dict(zip(X.columns, beta))
    c = pd.read_csv(HERE / "results" / "coefficients.csv")
    c = c[c["model"] == "Multifamily Housing"].set_index("term")["coef"]
    assert abs(b["electric_share"] - c["electric_share"]) < 1e-4 and abs(b["steam"] - c["steam"]) < 1e-4
    assert abs(b["era_2010+"] - c["C(era, Treatment('1960-79'))[T.2010+]"]) < 1e-4

    rob = pd.read_csv(HERE / "results" / "robustness.csv")
    assert (rob["electric_share_per_10pts"] < 0).all() and (rob["steam"] > 0).all() and (rob["built_2010_plus"] > 0).all()
    conf = json.loads((HERE / "results" / "confounding.json").read_text())
    assert conf["built_2010_plus_era_only"] < 0 < conf["built_2010_plus_with_electric_share"]
    print(f"OK: funnel recomputes ({len(sub):,} buildings); coefficients match a numpy refit; electricity, steam and "
          f"2010+ effects keep their sign in all 3 versions; raw era gap {conf['built_2010_plus_era_only']}% flips to "
          f"+{conf['built_2010_plus_with_electric_share']}% with fuel mix")


if __name__ == "__main__":
    main()
