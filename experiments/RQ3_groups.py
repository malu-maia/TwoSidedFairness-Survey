"""Evaluate whether the group fairness methods attend to underserved groups.

For Amazon, MovieLens, and Yelp, underserved consumers are the bottom 80% by
training activity and underserved providers are long-tail providers. For
AMBAR, female consumers and female artist-providers are protected using the
gender attributes in the raw dataset.

Provider attention is cumulative log-discounted exposure. Consumer utility is
the sum of held-out per-user NDCG. Both metrics are compared on the same run
before and after the fair method.

Run from the repository root with:

    python -m experiments.RQ3_groups

Select one or more prepared datasets by repeating ``--dataset``, and one or
more group fairness methods by repeating ``--method``:

    python -m experiments.RQ3_groups \
        --dataset amazon --dataset ambar \
        --dataset movielens --dataset yelp \
        --method cpfair --method tfld-groups

For each dataset and method, the report and comparison table are saved under
``results/RQ3_groups/<dataset>/<method>/``. The baseline and fair top-k
rankings for each run are saved in nested run directories.
"""

from __future__ import annotations

import argparse
import io
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence

import pandas as pd
import torch

from . import grouped_experiments as experiment_pipeline
from .metrics import ndcg


DEFAULT_RUNS_ROOT = experiment_pipeline.PROJECT_ROOT / "experiments" / "runs"
DEFAULT_RESULTS_ROOT = experiment_pipeline.PROJECT_ROOT / "results"
SUPPORTED_DATASETS = ("amazon", "ambar", "movielens", "yelp")
SUPPORTED_METHODS = ("cpfair", "tfld-groups")
METHOD_LABELS = {"cpfair": "CPFair", "tfld-groups": "TFLD-Groups"}
# Adapter names, not the functions themselves, so the lookup stays late-bound
# and patchable on the pipeline module.
_METHOD_RUNNERS = {"cpfair": "_run_cpfair", "tfld-groups": "_run_tfld_groups"}
UNDERSERVED_PROVIDER_GROUP = 1
SHORT_HEAD_FRACTION = 0.2
COMPARISON_TOLERANCE = 1e-12

# The AMBAR gender groups live in the experiment pipeline so grouped_experiments
# can use them too; re-exported here for callers and tests.
AMBAR_RAW_DIRECTORY = experiment_pipeline.AMBAR_RAW_DIRECTORY
PROTECTED_GROUP = experiment_pipeline.PROTECTED_GROUP
ADVANTAGED_GROUP = experiment_pipeline.ADVANTAGED_GROUP
FEMALE_LABELS = experiment_pipeline.FEMALE_LABELS
_validate_cpfair_groups = experiment_pipeline._validate_cpfair_groups
_ambar_cpfair_groups = experiment_pipeline._ambar_cpfair_groups


@dataclass(frozen=True)
class RunComparison:
    """Paired underserved-group measurements for one prepared run."""

    dataset: str
    seed: int
    run: int
    k: int
    provider_group_label: str
    underserved_provider_count: int
    provider_exposure_before: float
    provider_exposure_after: float
    providers_improved_fraction: float
    consumer_group_label: str
    underserved_consumer_count: int
    consumer_ndcg_before: float
    consumer_ndcg_after: float
    consumers_improved_fraction: float
    method: str = "cpfair"

    @property
    def provider_exposure_change(self) -> float:
        """Return the paired underserved-provider exposure change."""

        return self.provider_exposure_after - self.provider_exposure_before

    @property
    def consumer_ndcg_change(self) -> float:
        """Return the paired underserved-consumer summed-NDCG change."""

        return self.consumer_ndcg_after - self.consumer_ndcg_before


@dataclass(frozen=True)
class RunAnalysis:
    """Metrics and rankings produced by one paired fair-method analysis."""

    comparison: RunComparison
    topk_before: torch.Tensor
    topk_after: torch.Tensor


@dataclass(frozen=True)
class RankingFiles:
    """Paths of the persisted before-and-after rankings."""

    before: Path
    after: Path


@dataclass(frozen=True)
class ReportFiles:
    """Paths of the persisted report and per-run comparison table."""

    report: Path
    comparison: Path


@dataclass(frozen=True)
class AggregateComparison:
    """Macro-average of paired measurements across prepared runs."""

    run_count: int
    method: str
    provider_group_label: str
    provider_exposure_before: float
    provider_exposure_after: float
    provider_runs_improved: int
    providers_improved_fraction: float
    consumer_group_label: str
    consumer_ndcg_before: float
    consumer_ndcg_after: float
    consumer_runs_improved: int
    consumers_improved_fraction: float

    @property
    def provider_exposure_change(self) -> float:
        """Return the macro-average cumulative provider-exposure change."""

        return self.provider_exposure_after - self.provider_exposure_before

    @property
    def consumer_ndcg_change(self) -> float:
        """Return the macro-average summed consumer-NDCG change."""

        return self.consumer_ndcg_after - self.consumer_ndcg_before


def _first_existing(run_directory: Path, relative_paths: Sequence[str]) -> Path:
    candidates = [run_directory / relative_path for relative_path in relative_paths]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _validate_group_label(
    path: Path,
    *,
    group_column: str,
    group: int,
    expected_label: str,
) -> None:
    """Confirm that a numeric group still has the prepared-data semantics."""

    frame = experiment_pipeline._read_table(path)
    if group_column not in frame.columns:
        raise ValueError(f"{path} is missing required column {group_column!r}")
    if "group_label" not in frame.columns:
        return
    labels = set(
        frame.loc[frame[group_column] == group, "group_label"]
        .dropna()
        .astype(str)
        .str.lower()
    )
    if labels and labels != {expected_label}:
        raise ValueError(
            f"{path} maps group {group} to {sorted(labels)}, "
            f"expected {expected_label!r}"
        )


def _activity_user_groups(
    run: experiment_pipeline.PreparedRun,
) -> dict[int, int]:
    """Return CPFair's deterministic top-20% versus bottom-80% activity split."""

    activity = torch.count_nonzero(run.train_matrix, dim=1).tolist()
    short_head_count = max(1, math.ceil(SHORT_HEAD_FRACTION * len(activity)))
    short_head = set(
        sorted(
            range(len(activity)),
            key=lambda user_id: (-activity[user_id], user_id),
        )[:short_head_count]
    )
    groups = {
        user_id: (
            ADVANTAGED_GROUP if user_id in short_head else PROTECTED_GROUP
        )
        for user_id in range(len(activity))
    }
    _validate_cpfair_groups(groups, name="activity-based consumer groups")
    return groups


def _protected_mask(
    groups: Mapping[int, int],
    entity_ids: Iterable[int],
    *,
    name: str,
) -> torch.Tensor:
    ids = list(entity_ids)
    missing_ids = [entity_id for entity_id in ids if entity_id not in groups]
    if missing_ids:
        raise ValueError(f"{name} does not cover IDs: {missing_ids[:5]}")
    mask = torch.as_tensor(
        [groups[entity_id] == PROTECTED_GROUP for entity_id in ids],
        dtype=torch.bool,
    )
    if not mask.any():
        raise ValueError(f"{name} contains no protected entities")
    return mask


def _provider_group_path(run_directory: Path) -> Path:
    return _first_existing(
        run_directory,
        (
            "groups/provider_group_mapping.csv",
            "method_inputs/provider_defaults/provider_group_mapping.csv",
        ),
    )


def _underserved_provider_mask(
    run: experiment_pipeline.PreparedRun,
    provider_ids: torch.Tensor,
) -> torch.Tensor:
    path = _provider_group_path(run.directory)
    frame = experiment_pipeline._read_table(path)
    required = {"provider_id", "provider_group"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    mapping = frame[["provider_id", "provider_group"]].copy()
    for column in ("provider_id", "provider_group"):
        mapping[column] = pd.to_numeric(mapping[column], errors="coerce")
        values = mapping[column]
        if values.isna().any() or not values.eq(values.round()).all():
            raise ValueError(f"{path} column {column!r} must contain integers")
        mapping[column] = values.astype("int64")
    if mapping["provider_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate provider_id values")
    if set(mapping["provider_group"]) != {0, 1}:
        raise ValueError(f"{path} must define provider groups 0 and 1")

    _validate_group_label(
        path,
        group_column="provider_group",
        group=UNDERSERVED_PROVIDER_GROUP,
        expected_label="long_tail",
    )
    group_by_provider = mapping.set_index("provider_id")["provider_group"]
    missing_providers = [
        provider_id
        for provider_id in provider_ids.tolist()
        if provider_id not in group_by_provider.index
    ]
    if missing_providers:
        raise ValueError(
            f"{path} does not cover recommended-item providers: "
            f"{missing_providers[:5]}"
        )
    mask = torch.as_tensor(
        [
            int(group_by_provider.loc[provider_id])
            == UNDERSERVED_PROVIDER_GROUP
            for provider_id in provider_ids.tolist()
        ],
        dtype=torch.bool,
    )
    if not mask.any():
        raise ValueError(f"{path} contains no long-tail providers")
    return mask


def _provider_exposure(
    run: experiment_pipeline.PreparedRun,
    rankings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    item_exposure = experiment_pipeline._exposure(
        rankings,
        num_items=run.scores.shape[1],
        k=run.k,
    )
    return experiment_pipeline._provider_values_from_item_values(
        item_exposure,
        run.provider_ids,
    )


def _mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty underserved group")
    return float(values.to(torch.float64).mean().item())


def _sum(values: torch.Tensor) -> float:
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty underserved group")
    return float(values.to(torch.float64).sum().item())


def analyze_run(
    run: experiment_pipeline.PreparedRun,
    *,
    method: str = "cpfair",
    config: experiment_pipeline.RunnerConfig | None = None,
    device: str = "cpu",
) -> RunAnalysis:
    """Compare underserved providers and consumers around one fair-method run."""

    dataset = run.dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"{run.directory} belongs to unsupported dataset {run.dataset!r}; "
            f"choose from {list(SUPPORTED_DATASETS)}"
        )
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"unsupported method {method!r}; choose from {list(SUPPORTED_METHODS)}"
        )
    if dataset == "ambar":
        # Both methods switch AMBAR to gender groups themselves.
        user_groups, _, provider_groups = _ambar_cpfair_groups(run)
        underserved_users = _protected_mask(
            user_groups,
            range(run.scores.shape[0]),
            name="AMBAR consumer gender groups",
        )
        provider_group_label = "female artist"
        consumer_group_label = "female"
    else:
        user_groups = _activity_user_groups(run)
        underserved_users = _protected_mask(
            user_groups,
            range(run.scores.shape[0]),
            name="activity-based consumer groups",
        )
        provider_groups = None
        provider_group_label = "long-tail"
        consumer_group_label = "long-tail activity"

    runner = getattr(experiment_pipeline, _METHOD_RUNNERS[method])
    output = runner(
        run,
        config
        if config is not None
        else experiment_pipeline.RunnerConfig(device=device),
    )

    provider_ids, exposure_before = _provider_exposure(run, run.base_topk)
    fair_provider_ids, exposure_after = _provider_exposure(run, output.topk)
    if not torch.equal(provider_ids, fair_provider_ids):
        raise RuntimeError("provider universe changed after the fair method")
    underserved_providers = (
        _protected_mask(
            provider_groups,
            provider_ids.tolist(),
            name="AMBAR provider gender groups",
        )
        if provider_groups is not None
        else _underserved_provider_mask(run, provider_ids)
    )
    underserved_exposure_before = exposure_before[underserved_providers]
    underserved_exposure_after = exposure_after[underserved_providers]

    utility_before = ndcg(
        run.heldout_relevance,
        run.base_topk,
        k=run.k,
    ).values.to(torch.float64)
    utility_after = ndcg(
        run.heldout_relevance,
        output.topk,
        k=run.k,
    ).values.to(torch.float64)
    underserved_utility_before = utility_before[underserved_users]
    underserved_utility_after = utility_after[underserved_users]

    comparison = RunComparison(
        dataset=dataset,
        method=method,
        seed=run.seed,
        run=run.run,
        k=run.k,
        provider_group_label=provider_group_label,
        underserved_provider_count=int(underserved_providers.sum().item()),
        provider_exposure_before=_sum(underserved_exposure_before),
        provider_exposure_after=_sum(underserved_exposure_after),
        providers_improved_fraction=_mean(
            (
                underserved_exposure_after
                > underserved_exposure_before + COMPARISON_TOLERANCE
            ).to(torch.float64)
        ),
        consumer_group_label=consumer_group_label,
        underserved_consumer_count=int(underserved_users.sum().item()),
        consumer_ndcg_before=_sum(underserved_utility_before),
        consumer_ndcg_after=_sum(underserved_utility_after),
        consumers_improved_fraction=_mean(
            (
                underserved_utility_after
                > underserved_utility_before + COMPARISON_TOLERANCE
            ).to(torch.float64)
        ),
    )
    return RunAnalysis(
        comparison=comparison,
        topk_before=run.base_topk.detach().cpu().to(torch.long),
        topk_after=output.topk.detach().cpu().to(torch.long),
    )


def _topk_frame(rankings: torch.Tensor) -> pd.DataFrame:
    """Create an explicit user-by-rank table of mapped item IDs."""

    topk = rankings.detach().cpu().to(torch.long)
    if topk.ndim != 2 or topk.numel() == 0:
        raise ValueError("top-k rankings must be a non-empty two-dimensional tensor")
    columns = [f"rank_{rank}_item_id" for rank in range(1, topk.shape[1] + 1)]
    frame = pd.DataFrame(topk.numpy(), columns=columns)
    frame.insert(0, "user_id", range(topk.shape[0]))
    return frame


def save_topk_rankings(
    analysis: RunAnalysis,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> RankingFiles:
    """Save baseline and fair rankings to separate Parquet files."""

    result = analysis.comparison
    root = Path(results_root)
    if not root.is_absolute():
        root = experiment_pipeline.PROJECT_ROOT / root
    output_directory = (
        root
        / "RQ3_groups"
        / result.dataset
        / result.method
        / f"run_{result.run:02d}"
        / f"seed_{result.seed}"
        / f"k_{result.k}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    files = RankingFiles(
        before=output_directory / "topk_before.parquet",
        after=output_directory / "topk_after.parquet",
    )
    _topk_frame(analysis.topk_before).to_parquet(files.before, index=False)
    _topk_frame(analysis.topk_after).to_parquet(files.after, index=False)
    return files


def _average(values: Iterable[float]) -> float:
    return float(fmean(values))


def summarize(comparisons: Sequence[RunComparison]) -> AggregateComparison:
    """Macro-average paired results so each experimental run has equal weight."""

    if not comparisons:
        raise ValueError("at least one run comparison is required")
    datasets = {result.dataset for result in comparisons}
    if len(datasets) != 1:
        raise ValueError("run comparisons must belong to one dataset")
    methods = {result.method for result in comparisons}
    if len(methods) != 1:
        raise ValueError("run comparisons must belong to one method")
    provider_labels = {result.provider_group_label for result in comparisons}
    consumer_labels = {result.consumer_group_label for result in comparisons}
    if len(provider_labels) != 1 or len(consumer_labels) != 1:
        raise ValueError("run comparisons must use consistent underserved groups")
    return AggregateComparison(
        run_count=len(comparisons),
        method=next(iter(methods)),
        provider_group_label=next(iter(provider_labels)),
        provider_exposure_before=_average(
            result.provider_exposure_before for result in comparisons
        ),
        provider_exposure_after=_average(
            result.provider_exposure_after for result in comparisons
        ),
        provider_runs_improved=sum(
            result.provider_exposure_change > COMPARISON_TOLERANCE
            for result in comparisons
        ),
        providers_improved_fraction=_average(
            result.providers_improved_fraction for result in comparisons
        ),
        consumer_group_label=next(iter(consumer_labels)),
        consumer_ndcg_before=_average(
            result.consumer_ndcg_before for result in comparisons
        ),
        consumer_ndcg_after=_average(
            result.consumer_ndcg_after for result in comparisons
        ),
        consumer_runs_improved=sum(
            result.consumer_ndcg_change > COMPARISON_TOLERANCE
            for result in comparisons
        ),
        consumers_improved_fraction=_average(
            result.consumers_improved_fraction for result in comparisons
        ),
    )


def discover_dataset_runs(runs_root: str | Path, dataset: str) -> list[Path]:
    """Return prepared run directories for one dataset in deterministic order."""

    root = Path(runs_root).resolve()
    normalized_dataset = dataset.lower()
    if normalized_dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"unsupported dataset {dataset!r}; choose from {list(SUPPORTED_DATASETS)}"
        )
    dataset_root = (
        root if root.name.lower() == normalized_dataset else root / normalized_dataset
    )
    directories = sorted(
        path for path in dataset_root.glob("run_*/seed_*") if path.is_dir()
    )
    if not directories:
        raise FileNotFoundError(
            f"no prepared {normalized_dataset} runs found under {dataset_root}. "
            "Prepare them with: python -m "
            "experiments.prepare_explicit_method_inputs "
            f"--datasets {normalized_dataset}"
        )
    return directories


def _percent_change(before: float, after: float) -> float:
    if math.isclose(before, 0.0, abs_tol=COMPARISON_TOLERANCE):
        return math.nan
    return 100.0 * (after - before) / before


def _comparison_frame(comparisons: Sequence[RunComparison]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": result.method,
                "run": result.run,
                "seed": result.seed,
                "k": result.k,
                "provider_group": result.provider_group_label,
                "underserved_providers": result.underserved_provider_count,
                "provider_cumulative_exposure_before": (
                    result.provider_exposure_before
                ),
                "provider_cumulative_exposure_after": (
                    result.provider_exposure_after
                ),
                "provider_exposure_change_pct": _percent_change(
                    result.provider_exposure_before,
                    result.provider_exposure_after,
                ),
                "providers_improved_pct": (
                    100.0 * result.providers_improved_fraction
                ),
                "consumer_group": result.consumer_group_label,
                "underserved_consumers": result.underserved_consumer_count,
                "consumer_ndcg_sum_before": result.consumer_ndcg_before,
                "consumer_ndcg_sum_after": result.consumer_ndcg_after,
                "consumer_ndcg_sum_change_pct": _percent_change(
                    result.consumer_ndcg_before,
                    result.consumer_ndcg_after,
                ),
                "consumers_improved_pct": (
                    100.0 * result.consumers_improved_fraction
                ),
            }
            for result in comparisons
        ]
    )


def _direction(change: float) -> str:
    if change > COMPARISON_TOLERANCE:
        return "increased"
    if change < -COMPARISON_TOLERANCE:
        return "decreased"
    return "did not measurably change"


def research_answer(aggregate: AggregateComparison) -> str:
    """Return the descriptive answer supported by the paired comparisons."""

    providers_benefited = (
        aggregate.provider_exposure_change > COMPARISON_TOLERANCE
    )
    consumers_benefited = aggregate.consumer_ndcg_change > COMPARISON_TOLERANCE
    if providers_benefited and consumers_benefited:
        answer = "Yes, for both underserved providers and consumers."
    elif providers_benefited:
        answer = "Partially: underserved providers benefited, but consumers did not."
    elif consumers_benefited:
        answer = "Partially: underserved consumers benefited, but providers did not."
    else:
        answer = "No: neither underserved providers nor consumers benefited on average."
    return (
        f"{answer} {aggregate.provider_group_label.capitalize()} providers' "
        "cumulative exposure "
        f"{_direction(aggregate.provider_exposure_change)}, while "
        f"{aggregate.consumer_group_label} consumers' summed held-out NDCG "
        f"{_direction(aggregate.consumer_ndcg_change)}."
    )


def _report_text(comparisons: Sequence[RunComparison]) -> str:
    """Format per-run evidence, aggregate changes, and the research answer."""

    aggregate = summarize(comparisons)
    dataset = comparisons[0].dataset
    provider_change_pct = _percent_change(
        aggregate.provider_exposure_before,
        aggregate.provider_exposure_after,
    )
    consumer_change_pct = _percent_change(
        aggregate.consumer_ndcg_before,
        aggregate.consumer_ndcg_after,
    )
    output = io.StringIO()
    print(
        "RQ: Do the group fairness methods attend to the underserved groups?",
        file=output,
    )
    print(
        f"Method: {METHOD_LABELS[aggregate.method]} | Dataset: {dataset.upper()}",
        file=output,
    )
    print(
        f"Operational definition: {aggregate.provider_group_label} provider "
        "attention is cumulative log-discounted exposure; "
        f"{aggregate.consumer_group_label} consumer utility is summed held-out NDCG.",
        file=output,
    )
    print("\nPaired results by run:", file=output)
    print(
        _comparison_frame(comparisons).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        file=output,
    )
    print("\nMacro-average across runs:", file=output)
    print(
        f"Providers ({aggregate.provider_group_label}): cumulative exposure "
        f"{aggregate.provider_exposure_before:.6f} -> "
        f"{aggregate.provider_exposure_after:.6f} "
        f"({provider_change_pct:+.2f}%); "
        f"improved in {aggregate.provider_runs_improved}/{aggregate.run_count} runs; "
        f"{100.0 * aggregate.providers_improved_fraction:.2f}% of underserved "
        "providers improved per run on average.",
        file=output,
    )
    print(
        f"Consumers ({aggregate.consumer_group_label}): summed NDCG "
        f"{aggregate.consumer_ndcg_before:.6f} -> "
        f"{aggregate.consumer_ndcg_after:.6f} "
        f"({consumer_change_pct:+.2f}%); "
        f"improved in {aggregate.consumer_runs_improved}/{aggregate.run_count} runs; "
        f"{100.0 * aggregate.consumers_improved_fraction:.2f}% of underserved "
        "consumers improved per run on average.",
        file=output,
    )
    print(f"\nAnswer: {research_answer(aggregate)}", file=output)
    print(
        "Interpretation is descriptive: a positive paired change counts as "
        "attention to the underserved group; no significance test is implied.",
        file=output,
    )
    return output.getvalue()


def print_report(comparisons: Sequence[RunComparison]) -> None:
    """Print per-run evidence, aggregate changes, and the research answer."""

    print(_report_text(comparisons), end="")


def save_report_artifacts(
    comparisons: Sequence[RunComparison],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> ReportFiles:
    """Save one dataset's report and per-run comparison table."""

    report = _report_text(comparisons)
    dataset = comparisons[0].dataset
    root = Path(results_root)
    if not root.is_absolute():
        root = experiment_pipeline.PROJECT_ROOT / root
    output_directory = root / "RQ3_groups" / dataset / comparisons[0].method
    output_directory.mkdir(parents=True, exist_ok=True)
    files = ReportFiles(
        report=output_directory / "report.txt",
        comparison=output_directory / "comparison.tsv",
    )
    files.report.write_text(report, encoding="utf-8")
    _comparison_frame(comparisons).to_csv(
        files.comparison,
        sep="\t",
        index=False,
    )
    return files


def print_saved_rankings(files: Sequence[RankingFiles]) -> None:
    """Print the two ranking artifacts written for every analyzed run."""

    print("\nSaved top-k rankings:")
    for artifacts in files:
        print(f"Before: {artifacts.before}")
        print(f"After:  {artifacts.after}")


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Prepared runs root, or a selected dataset directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        help="Analyze one prepared seed directory; repeat as needed.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=SUPPORTED_DATASETS,
        help=(
            "Prepared dataset to analyze; repeat as needed "
            "(default: ambar)."
        ),
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=SUPPORTED_METHODS,
        help=(
            "Group fairness method to analyze; repeat as needed "
            "(default: cpfair)."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory for the saved before-and-after top-k files.",
    )
    parser.add_argument(
        "--tfld-epochs",
        type=_positive_integer,
        default=200,
        help="Frank-Wolfe iterations used by TFLD-Groups (default: 200).",
    )
    parser.add_argument(
        "--tfld-alpha",
        type=float,
        nargs=2,
        metavar=("ALPHA_USER", "ALPHA_ITEM"),
        default=(0.0, 0.0),
        help=(
            "TFLD-Groups welfare curvatures; both must be below 1. Smaller "
            "values focus harder on the worse-off group (default: 0.0 0.0)."
        ),
    )
    parser.add_argument(
        "--tfld-lambda",
        type=float,
        default=0.5,
        help="TFLD-Groups item-versus-user weight in [0, 1] (default: 0.5).",
    )
    parser.add_argument(
        "--k",
        type=_positive_integer,
        default=150,
        help="Recommendation-list size (default: 150).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used by the fair method, for example cpu or cuda.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the underserved-group analysis for the selected datasets and methods."""

    parser = build_parser()
    args = parser.parse_args(argv)
    selected_datasets = list(dict.fromkeys(args.dataset or ("ambar",)))
    selected_dataset_set = set(selected_datasets)
    selected_methods = list(dict.fromkeys(args.method or ("cpfair",)))
    config = experiment_pipeline.RunnerConfig(
        device=args.device,
        tfld_epochs=args.tfld_epochs,
        tfld_alpha=tuple(args.tfld_alpha),
        tfld_lambda=args.tfld_lambda,
    )
    try:
        directories = (
            [path.resolve() for path in args.run_dir]
            if args.run_dir
            else [
                directory
                for dataset in selected_datasets
                for directory in discover_dataset_runs(args.runs_root, dataset)
            ]
        )
        prepared_runs = [
            experiment_pipeline._load_prepared_run(
                directory,
                recommendation_list_size=args.k,
            )
            for directory in directories
        ]
        for run in prepared_runs:
            if run.dataset.lower() not in selected_dataset_set:
                raise ValueError(
                    f"{run.directory} belongs to dataset {run.dataset!r}, "
                    f"not one of the selected datasets {selected_datasets}"
                )
        analyses = [
            analyze_run(run, method=method, config=config)
            for method in selected_methods
            for run in prepared_runs
        ]
        ranking_files = [
            save_topk_rankings(analysis, args.results_root)
            for analysis in analyses
        ]
        dataset_comparisons = []
        for dataset in selected_datasets:
            for method in selected_methods:
                comparisons = [
                    analysis.comparison
                    for analysis in analyses
                    if analysis.comparison.dataset == dataset
                    and analysis.comparison.method == method
                ]
                if comparisons:
                    save_report_artifacts(comparisons, args.results_root)
                    dataset_comparisons.append(comparisons)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    for comparisons in dataset_comparisons:
        print_report(comparisons)
    print_saved_rankings(ranking_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
