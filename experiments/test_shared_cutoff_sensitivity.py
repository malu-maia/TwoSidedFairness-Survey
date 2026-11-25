"""Regression tests for input routing and validation-only selection."""

from __future__ import annotations

from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from . import grouped_experiments as survey
from . import sensitivity_inputs as inputs
from . import shared_cutoff_sensitivity as study
from .experiments import _fit_mf_by_cutoff, _matrix, _split
from obm._welfare import ranking_distribution, sorted_prefix, update_exposure


class SharedCutoffTest(unittest.TestCase):
    def test_respects_existing_pair_assignments(self):
        grouped = pd.DataFrame([(user, item, 5.0) for user in range(12) for item in range(6)],
                               columns=["user_id", "item_id", "rating"])
        standard = pd.concat([grouped, grouped.assign(item_id=lambda rows: rows.item_id + 6)], ignore_index=True)

        def labels(splits):
            return pd.concat([
                split[["user_id", "item_id"]].astype(str).assign(split=label)
                for label, split in enumerate(splits)
            ], ignore_index=True)

        fixed = labels(_split(grouped, 0.1, 0.1, 42))
        result = labels(_split(standard, 0.1, 0.1, 43, fixed))
        inputs.verify_shared_split(fixed, result, "fixture")
        self.assertFalse(result.duplicated(["user_id", "item_id"]).any())
        self.assertEqual(result.groupby("split").user_id.nunique().tolist(), [12, 12, 12])

    def test_all_methods_route_to_the_manifest_input(self):
        self.assertEqual(len(study.METHODS), 10)
        expected = {"amazon": "brand", "ambar": "standard", "movielens": "director-gender",
                    "yelp": "cpfair-territorial"}
        config = survey.RunnerConfig(frozen_settings=True)
        for dataset in inputs.DATASETS:
            for method in study.METHODS:
                group_method = method in inputs.GROUP_METHODS
                variant = expected[dataset] if group_method else "standard"
                self.assertEqual(inputs.variant_for(dataset, method), variant)
                if group_method:
                    self.assertEqual(survey._group_partition(dataset, method, config) or "standard", variant)

    def test_validation_cohort_uses_no_test_relevance(self):
        data = torch.tensor([[0., 0., 1.], [0., 1., 0.]])
        parent = survey.PreparedRun(
            directory=Path("fixture"), dataset="amazon", seed=42, run=0, k=0, scores=data,
            train_matrix=data, heldout_relevance=torch.tensor([[1., 0., 0.], [1., 0., 0.]]),
            eligible_items=data == 0, base_topk=torch.empty(0), item_merit=torch.ones(3),
            provider_ids=torch.arange(3),
        )
        with patch.object(study, "CUTOFFS", (1, 2)), patch.object(study, "_frame_matrix") as read:
            cohort, _, users = study._cohort(parent, grouped=False, split="validation")
            self.assertEqual(users.tolist(), [0, 1])
            self.assertTrue(torch.equal(cohort.heldout_relevance, parent.heldout_relevance))
            read.assert_not_called()

    def test_votes_keep_failed_pairs_and_prefer_shorter_cutoff_on_tie(self):
        rows = []
        for dataset in inputs.DATASETS:
            for method in study.METHODS:
                for k in (10, 25):
                    for seed in inputs.FIT_SEEDS:
                        rows.append({
                            "dataset": dataset, "method": method, "setting": "ref", "k": k, "seed": seed,
                            "control": False, "reference": True, "ndcg_margin": 0.0,
                            "ndcg_delta": -0.02, "consumer_gain": 0.05, "supplier_gain": 0.10,
                            "variant": "standard", "parameters": "{}", "error": "",
                            "status": "quota_violated" if method == "SEAL" else "ok",
                        })
        votes = study._cutoff_votes(study._summarize(pd.DataFrame(rows)))
        self.assertEqual(votes.k.tolist(), [10, 25])
        self.assertEqual(votes.method_dataset_pairs.tolist(), [40, 40])
        self.assertEqual(votes.votes.tolist(), [36, 36])

    def test_checkpoints_are_selected_independently_with_earliest_ties(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scores", torch.empty((1, 6)))

            def matrix_for_ranking(self):
                return self.scores

        def epochs(*args, **kwargs):
            model = Model()
            for epoch, values in enumerate(([5, 0, 4, 3, 2, 1], [4, 3, 5, 2, 1, 0], [4, 3, 5, 2, 1, 0]), 1):
                model.scores.copy_(torch.tensor([values]))
                yield epoch, model

        columns = ["user_id", "item_id", "rating"]
        train = pd.DataFrame([(0, 5, 1.0)], columns=columns)
        validation = pd.DataFrame([(0, 0, 1.0), (0, 1, 1.0)], columns=columns)
        options = study.BASELINE | {"epochs": 3}
        with patch("experiments.experiments._mf_epochs", side_effect=epochs) as fit:
            selected = list(_fit_mf_by_cutoff(train, validation, 1, 6, cutoffs=(1, 3), seed=42, **options))
        fit.assert_called_once()
        self.assertEqual([(k, epoch) for k, _, epoch, _ in selected], [(1, 1), (3, 2)])
        self.assertEqual([scores.argmax() for _, scores, _, _ in selected], [0, 2])

    def test_selection_prefers_valid_positive_seal_then_uses_zero_minimum(self):
        rows = pd.DataFrame([
            {"setting": "min_1_max_users", "reference": True, "product_min": 10, "eligible": False,
             "successful_seeds": 0, "weaker_gain": 0.5, "ndcg_delta": 0.0, "control": False},
            {"setting": "min_0.25_max_users", "reference": False, "product_min": 2, "eligible": False,
             "successful_seeds": 3, "weaker_gain": 0.1, "ndcg_delta": -0.2, "control": False},
            {"setting": "min_0_max_users", "reference": False, "product_min": 0, "eligible": True,
             "successful_seeds": 3, "weaker_gain": 0.9, "ndcg_delta": 0.0, "control": False},
        ])
        winner, reason = study._select_setting(rows, "SEAL")
        self.assertEqual((winner.setting, reason), ("min_0.25_max_users", "valid_below_ndcg_fallback"))
        rows.loc[1, "successful_seeds"] = 2
        winner, reason = study._select_setting(rows, "SEAL")
        self.assertEqual((winner.setting, reason), ("min_0_max_users", "quota_disabled_fallback"))
        rows.loc[2, "successful_seeds"] = 0
        winner, reason = study._select_setting(rows, "SEAL")
        self.assertEqual((winner.setting, reason), ("min_1_max_users", "no_valid_setting"))

    def test_valid_reference_is_retained_below_ndcg_floor(self):
        rows = pd.DataFrame([
            {"setting": "reference", "reference": True, "eligible": False, "successful_seeds": 3,
             "weaker_gain": 0.01, "ndcg_delta": -0.2, "control": False},
            {"setting": "alternative", "reference": False, "eligible": False, "successful_seeds": 3,
             "weaker_gain": 0.8, "ndcg_delta": -0.1, "control": False},
        ])
        winner, reason = study._select_setting(rows, "FairSort")
        self.assertEqual((winner.setting, reason), ("reference", "reference_fallback"))

    def test_phases_use_frozen_settings_and_keep_all_seed_rows(self):
        class InlinePool:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def submit(self, fn, *args):
                future = Future()
                try:
                    future.set_result(fn(*args))
                except Exception as error:
                    future.set_exception(error)
                return future

        with tempfile.TemporaryDirectory() as location:
            root = Path(location)
            results = root / "results"
            results.mkdir()
            run = inputs.input_directory(root, "amazon", "standard")
            run.mkdir(parents=True)
            for name, value in {
                "metadata.json": {"dataset": "amazon", "num_users": 6, "num_items": 6},
                "mappings.json": {}, "sensitivity_input.json": {},
            }.items():
                (run / name).write_text(json.dumps(value))
            train, validation, test = [
                pd.DataFrame([(user, (user + offset) % 6, 1.0) for user in range(6) for offset in offsets],
                             columns=["user_id", "item_id", "rating"])
                for offsets in [(0, 1), (2,), (3,)]
            ]
            train.to_parquet(run / "train_interactions.parquet")
            validation.to_parquet(run / "validation_interactions.parquet")
            pd.DataFrame(_matrix(train, 6, 6)).to_parquet(run / "train_matrix.parquet")
            pd.DataFrame({"item_id": range(6), "provider_id": range(6)}).to_csv(
                run / "provider_item_mapping.csv", index=False,
            )
            pd.DataFrame({"item_id": range(6), "merit": [2] * 6}).to_csv(run / "item_merit.csv", index=False)
            grid = {key: value for key, value in study.GRID.items() if value.method in {"SEAL", "FairRec"}}
            with (
                patch.object(inputs, "DATASETS", ("amazon",)),
                patch.object(study, "CUTOFFS", (1, 2)),
                patch.object(study, "GRID", grid),
                patch.object(study, "METHODS", ("FairRec", "SEAL")),
                patch.object(study, "BASELINE", study.BASELINE | {"epochs": 3}),
                patch.object(study, "ProcessPoolExecutor", InlinePool),
            ):
                study.run(root, results, 1, 1, "validation")
                self.assertFalse((results / "test_manifest.json").exists())
                test.to_parquet(run / "test_interactions.parquet")
                with patch.object(study, "_fit_mf_by_cutoff", side_effect=AssertionError("test refit")):
                    study.run(root, results, 1, 1, "test")
                checkpoint = study._checkpoint_dir(results, "amazon", "standard", 42) / "k_1.npy"
                with checkpoint.open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "checkpoint artifacts differ"):
                    study.run(root, results, 1, 1, "test")
            selected = pd.read_csv(results / "selected_settings.tsv", sep="\t")
            measured = pd.read_csv(results / "test_rq2_metrics.tsv", sep="\t")
            checkpoints = pd.read_csv(results / "checkpoints.tsv", sep="\t")
            self.assertEqual(len(measured), 12)
            self.assertEqual(len(selected), 4)
            self.assertEqual(len(pd.read_csv(results / "test_primary_by_dataset.tsv", sep="\t")), 2)
            self.assertEqual(
                set(measured[["method", "k", "setting"]].itertuples(index=False, name=None)),
                set(selected[["method", "k", "setting"]].itertuples(index=False, name=None)),
            )
            paired = measured.merge(checkpoints, on=["dataset", "variant", "seed", "k"], validate="many_to_one")
            self.assertTrue(paired.bpr_best_epoch.eq(paired.best_epoch).all())

    def test_frozen_settings_load_for_grouped_and_rq3(self):
        methods = {method: setting.manifest() for method, setting in study.REFERENCES.items()}
        document = {"format_version": 2, "k": 25, "datasets": {"amazon": methods}}
        with tempfile.TemporaryDirectory() as location:
            path = Path(location) / "settings.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            grouped, k = survey.load_hyperparameter_config(path, cpfair_suite="RQ4")
            rq3, _ = survey.load_hyperparameter_config(path, cpfair_suite="RQ3")
        self.assertEqual(k, 25)
        self.assertEqual(grouped, rq3)
        self.assertEqual(grouped["amazon"]["FairSort"], methods["FairSort"]["parameters"])
        self.assertEqual(set(grouped["amazon"]), set(study.METHODS))

    def test_grouped_runner_applies_each_setting_to_its_input(self):
        def fake_run(directory, _):
            return SimpleNamespace(dataset="amazon", directory=directory, seed=42)

        with tempfile.TemporaryDirectory() as location:
            root = Path(location)
            standard = root / "amazon/run_00/seed_42"
            grouped = root / "amazon/brand/run_00/seed_42"
            standard.mkdir(parents=True)
            grouped.mkdir(parents=True)
            config = survey.RunnerConfig(
                frozen_settings=True,
                best_hyperparams={"amazon": {"FairSort": {"rerank_ratio": 0.2}, "CPFair": {"lambda1": 0.5}}},
            )
            comparison = survey.GroupSpec(1, "fairness", "user", "individual", ("FairSort", "CPFair"), "metric")
            with (
                patch.object(survey, "_load_prepared_run", side_effect=fake_run),
                patch.object(survey, "_metadata", side_effect=lambda p: {"amazon_partition": "brand"} if p == grouped else {}),
                patch.object(survey, "_run_frozen_setting") as execute,
                patch.object(survey, "_result_row", return_value={"dataset": "amazon"}),
            ):
                survey.run_grouped_experiments(run_dirs=[standard, grouped], results_root=root / "results",
                                               groups=(comparison,), config=config)
        outcomes = {method: (run.directory, frozen.hyperparams_for("amazon", method))
                    for run, frozen, method in (call.args for call in execute.call_args_list)}
        self.assertEqual(outcomes, {
            "FairSort": (standard, {"rerank_ratio": 0.2}), "CPFair": (grouped, {"lambda1": 0.5}),
        })

    def test_faster_sort_preserves_prefix_even_when_scores_tie(self):
        generator = torch.Generator().manual_seed(42)
        scores = torch.rand((12, 100), generator=generator)
        scores[:4] = scores[:4].round(decimals=1)
        expected = scores.sort(dim=1, descending=True).indices
        for k in (1, 10, 50, 100):
            self.assertTrue(torch.equal(sorted_prefix(scores, k), expected[:, :k]))

    def test_expected_exposure_update_preserves_the_dense_step(self):
        rankings = torch.tensor([[0, 2], [1, 0]])
        discounts = torch.tensor([1.0, 0.6])
        exposure = torch.tensor([[0.34, 0.79, 0.67], [0.25, 0.03, 0.51]])
        direction = ranking_distribution(rankings, 3, discounts, exposure_only=True)
        step = 2 / 7
        expected = (1 - step) * exposure + step * direction
        self.assertTrue(torch.equal(update_exposure(exposure, rankings, discounts, step), expected))


if __name__ == "__main__":
    unittest.main()
