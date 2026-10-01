"""Step 1: NYC Local Law 84 energy benchmarking, buildings' 2024 reports (NYC Open Data 5zyy-y8am).

    ./venv/bin/python download.py

Large NYC buildings must report their energy and water use every year (Local Law 84). One row per property.
Only the columns this analysis uses are downloaded, into data/raw/ (not committed).
"""
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "data" / "raw" / "ll84_2024.csv"
COLS = ["property_id", "parent_property_id", "report_year", "primary_property_type", "largest_property_use_type",
        "year_built", "number_of_buildings", "occupancy", "construction_status", "energy_star_score",
        "site_eui_kbtu_ft", "weather_normalized_site_eui", "source_eui_kbtu_ft", "percent_electricity",
        "natural_gas_use_kbtu", "district_steam_use_kbtu", "fuel_oil_2_use_kbtu", "fuel_oil_4_use_kbtu",
        "fuel_oil_5_6_use_kbtu", "electricity_use_grid_purchase", "total_location_based_ghg", "total_location_based_ghg_1",
        "property_gfa_self_reported", "property_gfa_calculated", "multifamily_housing_total", "borough", "latitude", "longitude",
        "default_values", "estimated_data_flag", "alert_gross_floor_area_is", "alert_energy_meter_has_gaps",
        "alert_energy_no_meters"]

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    q = urllib.parse.urlencode({"$select": ",".join(COLS), "$where": "report_year=2024", "$limit": 60000})
    ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    OUT.write_bytes(urllib.request.urlopen(f"https://data.cityofnewyork.us/resource/5zyy-y8am.csv?{q}", context=ctx, timeout=300).read())
    print(OUT, sum(1 for _ in OUT.open()) - 1, "rows")
