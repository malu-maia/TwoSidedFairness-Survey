"""Map Yelp businesses to 2015–2019 ACS tract poverty estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def build_mapping(business_path: Path, tracts_path: Path, poverty_path: Path) -> pd.DataFrame:
    businesses = pd.read_json(business_path, lines=True, dtype={"business_id": str})
    required = {"business_id", "latitude", "longitude"}
    if missing := required - set(businesses):
        raise ValueError(f"{business_path} missing columns: {sorted(missing)}")
    if businesses.business_id.isna().any() or businesses.business_id.duplicated().any():
        raise ValueError("Yelp business IDs must be nonmissing and unique")
    tracts = gpd.read_file(tracts_path)
    if "GEOID" not in tracts or tracts.crs is None:
        raise ValueError(f"{tracts_path} must have GEOID and a coordinate system")
    if tracts.GEOID.duplicated().any():
        raise ValueError("Tract GEOIDs must be unique")
    tracts = tracts[["GEOID", "geometry"]].to_crs(4326)
    poverty = pd.read_csv(poverty_path, sep="|", usecols=["GEO_ID", "B17001_E001", "B17001_E002"], dtype=str)
    poverty = poverty[poverty.GEO_ID.str.startswith("1400000US", na=False)].copy()
    poverty["tract_id"] = poverty.GEO_ID.str.removeprefix("1400000US")
    if poverty.tract_id.duplicated().any():
        raise ValueError("Poverty data contain duplicate tract IDs")
    total = pd.to_numeric(poverty.B17001_E001, errors="coerce")
    poor = pd.to_numeric(poverty.B17001_E002, errors="coerce")
    valid_rate = (total > 0) & (poor >= 0) & (poor <= total)
    poverty["poverty_rate"] = (100 * poor / total).where(valid_rate)
    lat = pd.to_numeric(businesses.latitude, errors="coerce")
    lon = pd.to_numeric(businesses.longitude, errors="coerce")
    valid = lat.between(-90, 90) & lon.between(-180, 180)
    points = gpd.GeoDataFrame(businesses.loc[valid, ["business_id"]], geometry=gpd.points_from_xy(lon[valid], lat[valid]), crs=4326)
    matched = gpd.sjoin(points, tracts, how="inner", predicate="intersects")
    counts = matched.groupby("business_id").size()
    matched = matched[matched.business_id.map(counts).eq(1)][["business_id", "GEOID"]]
    result = businesses[["business_id"]].merge(matched, on="business_id", how="left", validate="one_to_one")
    result = result.rename(columns={"GEOID": "tract_id"}).merge(
        poverty[["tract_id", "poverty_rate"]], on="tract_id", how="left", validate="many_to_one"
    )
    result["match_status"] = "missing_poverty"
    result.loc[result.tract_id.isna(), "match_status"] = "unmatched_tract"
    result.loc[~valid, "match_status"] = "invalid_coordinates"
    result.loc[result.business_id.isin(counts[counts > 1].index), "match_status"] = "ambiguous_tract"
    result.loc[result.poverty_rate.notna(), "match_status"] = "matched"
    result["item_group"] = np.where(result.poverty_rate.notna(), np.where(result.poverty_rate >= 20, 1, -1), np.nan)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--businesses", type=Path, required=True)
    parser.add_argument("--tracts", type=Path, required=True)
    parser.add_argument("--poverty", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_mapping(args.businesses, args.tracts, args.poverty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    metadata = {
        "businesses": str(args.businesses), "tracts": str(args.tracts), "poverty": str(args.poverty),
        "tract_source": "https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_us_tract_500k.zip",
        "poverty_source": "https://www2.census.gov/programs-surveys/acs/summary_file/2019/prototype/5YRData/acsdt5y2019-b17001.dat",
        "poverty_measure": "ACS 2015-2019 B17001_E002 / B17001_E001 * 100",
        "threshold_percent": 20,
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(result.match_status.value_counts().to_string())
    print(result.item_group.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
