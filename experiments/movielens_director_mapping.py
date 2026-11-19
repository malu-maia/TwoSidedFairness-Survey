"""Join MovieLens 1M films to Wikidata directors through MovieLens 20M links."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MOVIES = ROOT / "data/raw_datasets/explicit_feedback/Movie Lens/movies.dat"
AUXILIARY = ROOT / "data/raw_datasets/explicit_feedback/Movie Lens/auxiliary_data"
LINKS = AUXILIARY / "links-20m.csv"
OUTPUT = AUXILIARY / "director_gender.csv"
ENDPOINT = "https://query.wikidata.org/sparql"


def query_directors(imdb_ids: list[str]) -> list[dict[str, str]]:
    values = " ".join(json.dumps(value) for value in imdb_ids)
    query = f"""SELECT ?imdb ?film ?director ?directorLabel ?gender ?genderLabel WHERE {{
      VALUES ?imdb {{ {values} }}
      ?film wdt:P345 ?imdb .
      OPTIONAL {{ ?film wdt:P57 ?director . OPTIONAL {{ ?director wdt:P21 ?gender . }} }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}"""
    request = Request(
        ENDPOINT, data=urlencode({"query": query, "format": "json"}).encode("utf-8"),
        headers={"User-Agent": "MovieLensDirectorResearch/1.0 (academic fairness research)",
                 "Accept": "application/sparql-results+json"},
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=90) as response:
                bindings = json.load(response)["results"]["bindings"]
            return [{key: value["value"] for key, value in row.items()} for row in bindings]
        except (HTTPError, URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 3:
                raise RuntimeError(f"Wikidata query failed after 4 attempts: {error}") from error
            delay = 65 if isinstance(error, HTTPError) and error.code == 429 else 2 ** (attempt + 1)
            print(f"Wikidata returned {error}; retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def build_mapping(movies_path: Path, links_path: Path, output: Path, batch_size: int) -> None:
    movies = pd.read_csv(movies_path, sep="::", engine="python", encoding="latin-1",
                         header=None, names=["movie_id", "title", "genres"])
    links = pd.read_csv(links_path, dtype={"imdbId": "string"})
    if movies.movie_id.duplicated().any() or links.movieId.duplicated().any():
        raise ValueError("MovieLens movie IDs must be unique in both input files")
    links["imdb_id"] = links.imdbId.map(lambda value: "tt" + value.zfill(7) if pd.notna(value) else None)
    films = movies[["movie_id", "title"]].merge(
        links[["movieId", "imdb_id"]], left_on="movie_id", right_on="movieId", how="left"
    ).drop(columns="movieId")
    ids = films.imdb_id.dropna().unique().tolist()
    cache_path = output.with_suffix(".checkpoint.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    remaining = [imdb_id for imdb_id in ids if imdb_id not in cache]
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        batch_rows = query_directors(batch)
        for imdb_id in batch:
            cache[imdb_id] = []
        for row in batch_rows:
            cache[row["imdb"]].append(row)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"Wikidata: {len(ids) - len(remaining) + min(start + batch_size, len(remaining))}/{len(ids)} IMDb IDs", flush=True)
        if start + batch_size < len(remaining):
            time.sleep(65)
    rows = [row for imdb_id in ids for row in cache[imdb_id]]
    columns = ["imdb", "film", "director", "directorLabel", "gender", "genderLabel"]
    records = pd.DataFrame(rows).reindex(columns=columns).rename(columns={
        "imdb": "imdb_id", "film": "wikidata_film_id", "director": "director_id",
        "directorLabel": "director_name", "gender": "gender_id", "genderLabel": "gender",
    })
    for column in ("wikidata_film_id", "director_id", "gender_id"):
        records[column] = records[column].str.rsplit("/", n=1).str[-1]
    result = films.merge(records, on="imdb_id", how="left")
    result["match_status"] = "matched"
    result.loc[result.gender_id.isna(), "match_status"] = "missing_gender"
    result.loc[result.director_id.isna(), "match_status"] = "missing_director"
    result.loc[result.wikidata_film_id.isna(), "match_status"] = "missing_wikidata_film"
    result.loc[result.imdb_id.isna(), "match_status"] = "missing_imdb_id"
    result = result.drop_duplicates().sort_values(["movie_id", "director_id", "gender_id"], na_position="last")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} rows for {result.movie_id.nunique()} films to {output}")
    print(result.groupby("match_status").movie_id.nunique().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies", type=Path, default=MOVIES)
    parser.add_argument("--links", type=Path, default=LINKS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    build_mapping(args.movies, args.links, args.output, args.batch_size)


if __name__ == "__main__":
    main()
