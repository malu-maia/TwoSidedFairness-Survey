"""Map Yelp businesses to official metropolitan boundaries supplied locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# Codes from the 2020 US CBSA and 2021 Canadian CMA classifications.
US_REGIONS = {
    "37980": "Philadelphia", "46060": "Tucson", "45300": "Tampa",
    "26900": "Indianapolis", "34980": "Nashville", "35380": "New Orleans",
    "39900": "Reno", "41180": "St. Louis", "42200": "Santa Barbara",
    "14260": "Boise",
}
REGION_CODES = {name: i for i, name in enumerate([*US_REGIONS.values(), "Edmonton"])}
UNKNOWN = "unmatched"


def load_boundaries(us_path: Path, canada_path: Path) -> gpd.GeoDataFrame:
    """Read the selected US CBSAs and Edmonton CMA, preserving their IDs."""
    us = gpd.read_file(us_path)
    canada = gpd.read_file(canada_path)
    for frame, field, path in ((us, "GEOID", us_path), (canada, "CMAUID", canada_path)):
        if field not in frame or frame.crs is None:
            raise ValueError(f"{path}: expected {field} and a defined coordinate system")
    us = us.assign(region=us["GEOID"].astype(str).map(US_REGIONS)).dropna(subset=["region"])
    canada = canada.loc[canada["CMAUID"].astype(str) == "835"].copy()
    canada["region"] = "Edmonton"
    boundaries = gpd.GeoDataFrame(
        pd.concat([us[["region", "geometry"]].to_crs(4326),
                   canada[["region", "geometry"]].to_crs(4326)], ignore_index=True),
        crs="EPSG:4326",
    )
    missing = set(REGION_CODES) - set(boundaries["region"])
    if missing:
        raise ValueError(f"Boundary files are missing regions: {', '.join(sorted(missing))}")
    if boundaries.geometry.isna().any() or boundaries.geometry.is_empty.any() or not boundaries.geometry.is_valid.all():
        raise ValueError("Boundary files contain missing, empty, or invalid geometries")
    return boundaries.dissolve(by="region", as_index=False)


def map_businesses(business_path: Path, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
    """Assign only unambiguous polygon intersections; retain every business."""
    businesses = pd.read_json(business_path, lines=True, dtype={"business_id": str})
    required = {"business_id", "latitude", "longitude"}
    if missing := required - set(businesses):
        raise ValueError(f"{business_path}: missing columns {sorted(missing)}")
    if businesses["business_id"].isna().any() or businesses["business_id"].eq("").any() or businesses["business_id"].duplicated().any():
        raise ValueError("Yelp business_id values must be present and unique")
    latitude = pd.to_numeric(businesses["latitude"], errors="coerce")
    longitude = pd.to_numeric(businesses["longitude"], errors="coerce")
    valid = latitude.between(-90, 90) & longitude.between(-180, 180)
    points = gpd.GeoDataFrame(
        businesses.loc[valid, ["business_id"]],
        geometry=gpd.points_from_xy(longitude[valid], latitude[valid]), crs="EPSG:4326",
    )
    matches = gpd.sjoin(points, boundaries, how="inner", predicate="intersects")
    counts = matches.groupby("business_id").size()
    unique = matches[matches["business_id"].map(counts).eq(1)].set_index("business_id")["region"]
    result = businesses[["business_id"]].copy()
    result["region"] = result["business_id"].map(unique).fillna(UNKNOWN)
    result["region_code"] = result["region"].map(REGION_CODES).fillna(-1).astype(int)
    result["match_status"] = np.where(~valid, "invalid_coordinates", "unmatched")
    result.loc[result["region"] != UNKNOWN, "match_status"] = "matched"
    result.loc[result["business_id"].isin(counts[counts > 1].index), "match_status"] = "ambiguous"
    return result


def map_run(mapping_path: Path, businesses: pd.DataFrame) -> pd.DataFrame:
    document = json.loads(mapping_path.read_text(encoding="utf-8"))
    items = document.get("items")
    if not isinstance(items, list) or any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{mapping_path}: items must be a list of Yelp business IDs")
    if len(items) != len(set(items)):
        raise ValueError(f"{mapping_path}: duplicate item IDs")
    result = pd.DataFrame({"item_id": range(len(items)), "business_id": items})
    result = result.merge(businesses, on="business_id", how="left", validate="one_to_one", sort=False)
    missing = result.loc[result["region"].isna(), "business_id"]
    if not missing.empty:
        raise ValueError(f"{mapping_path}: {len(missing)} items absent from Yelp businesses: {missing.head().tolist()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--businesses", type=Path, required=True)
    parser.add_argument("--us-boundaries", type=Path, required=True)
    parser.add_argument("--canada-boundaries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", default=[], help="Run directory containing mappings.json; repeat as needed")
    args = parser.parse_args()

    boundaries = load_boundaries(args.us_boundaries, args.canada_boundaries)
    businesses = map_businesses(args.businesses, boundaries)
    runs = [(path, map_run(path / "mappings.json", businesses)) for path in args.run]
    names = [f"{path.parent.name}_{path.name}" for path, _ in runs]
    if len(names) != len(set(names)):
        raise ValueError("Run output names collide; run each directory separately")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    businesses.to_csv(args.output_dir / "yelp_business_metro.csv", index=False)
    for name, (_, frame) in zip(names, runs, strict=True):
        frame.to_csv(args.output_dir / f"yelp_items_{name}.csv", index=False)
    metadata = {"businesses": str(args.businesses), "us_boundaries": str(args.us_boundaries),
                "canada_boundaries": str(args.canada_boundaries), "region_codes": REGION_CODES,
                "unknown_code": -1}
    (args.output_dir / "yelp_metro_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Businesses: {len(businesses)}")
    print(businesses["region"].value_counts().to_string())
    print("\nMatch status:")
    print(businesses["match_status"].value_counts().to_string())
    for name, (_, frame) in zip(names, runs, strict=True):
        print(f"{name}: {len(frame)} items")


if __name__ == "__main__":
    main()
