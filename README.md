# What drives energy use in NYC apartment buildings? (regression)

Under Local Law 84, every large NYC building reports its energy use each year. This models what goes with higher or
lower energy use intensity (EUI, kBtu per sq ft): construction era, fuel mix, district steam, size, density and borough.
It uses 10,490 apartment buildings' 2024 reports, with full regression diagnostics and robustness checks.

Build log: https://mrrishit909.github.io/projects/nyc-building-energy/

## Data

NYC Mayor's Office of Climate & Environmental Justice, "NYC Building Energy and Water Data Disclosure for Local Law 84
(2022–Present)", NYC Open Data `5zyy-y8am`, report year 2024 (39,090 properties). `download.py` fetches the columns used.

## Steps

| Step | File | What it does |
|---|---|---|
| 1 | `download.py` | the 2024 reports (not committed) |
| 2 | `analysis.py` | documented cleaning funnel; OLS on log weather-normalised site EUI; diagnostics; robustness; confounding check |
| 3 | `charts.py` | effects with confidence intervals, eras, residual diagnostics, electrification |
| 4 | `check.py` | funnel recomputes; coefficients match an independent NumPy refit; effects keep their sign in every version |

## Cleaning (multifamily)

39,090 reports → 38,857 existing buildings (230 "Test" and 3 "Design" records dropped) → 26,989 standalone properties (campus members
are reported again under their parent) → 18,427 multifamily → 16,867 with weather-normalised EUI. Then 16,720 within
10–500 kBtu/sq ft (values up to 11.7 million exist) and 16,547 at least 25,000 sq ft, the law's threshold. Finally 10,508
without estimated or default values, and **10,491** with an electricity share. Re-running the model with the estimated-value
buildings kept changes nothing important (see robustness).

## Results (multifamily, n = 10,490)

The model explains **59% of the variation in log EUI (cross-validated R² 0.588)**. Building size alone explains 0.2%.
Robust (HC3) standard errors are used because the residual spread varies (Breusch–Pagan p < 0.001). The largest
variance inflation factor is 3.5, and a Huber robust regression moves no coefficient by more than 0.06.

| Holding the others fixed | Effect on EUI |
|---|---:|
| 10 points more of the building's energy from electricity | **−15.9%** |
| Uses district steam | **+14.0%** |
| Built 2000–09 (vs 1960–79) | +13.1% |
| Built 2010 or later (vs 1960–79) | +11.4% |
| Built before 1930 / 1930–59 | −4.7% / −8.9% |
| Queens / Brooklyn / Bronx (vs Manhattan) | −12.0% / −8.0% / −5.0% |
| Burns fuel oil | +0.4% (not significant) |

**The lesson in confounding:** on raw medians, 2010+ buildings use 36% less energy than 1960–79 ones (51 vs 79 kBtu/sq ft).
But they are far more electric: the median electricity share is 52% against 19%. Hold the fuel mix equal and they use
**9% more** (11% in the full model). Comparing eras without the fuel mix gives the wrong answer.

**Robustness**

| Version | n | Electricity, per +10 pts | Steam | Built 2010+ |
|---|---:|---:|---:|---:|
| Main: clean set, site EUI | 10,490 | −15.9% | +14.0% | +11.4% |
| Keep estimated/default-value buildings | 16,505 | −14.9% | +12.4% | +11.7% |
| Source EUI (counts power-plant losses) | 10,490 | −7.3% | +21.5% | +13.0% |

Part of the electrification effect is accounting. Site energy counts a kWh at face value, while source energy charges
grid electricity for generation losses. On that basis the effect halves, but it stays negative. Offices are much harder
to explain (R² 0.23, CV 0.19); their electricity share (−56% across the full range) and occupancy matter most.

## Not done

- Not causal: buildings that electrified may differ in other ways (renovations, ownership).
- No Local Law 97 emissions-limit screen. That needs the law's own emission factors, not the reported location-based
  greenhouse-gas figures.
- One year only; a panel of 2022–2024 reports would show changes within buildings.
