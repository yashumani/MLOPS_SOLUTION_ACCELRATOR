"""Test holdout split written by stage4_feature_engineering.save_outputs."""

import ast

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from steps import stage3_preprocessing
from steps.stage2_preparation import extract_raw_train_and_holdout, prep_dataframe
from steps.stage3_preprocessing import preprocess
from steps.stage4_feature_engineering import feature_engineer
from steps.stage4_feature_engineering import save_outputs
from utils.holdout_partition import (
    HOLDOUT_PARTITION,
    ROW_ID_COLUMN,
    SPLIT_COLUMN,
    TRAIN_PARTITION,
    ensure_holdout_partition,
)


def _empty_report():
    return {"imbalance_metadata": {}}


def _dict_assignments(source: str, target_name: str) -> list[dict[str, str]]:
    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name) or node.value.func.id != "dict":
            continue
        matches.append(
            {
                keyword.arg: ast.unparse(keyword.value)
                for keyword in node.value.keywords
                if keyword.arg
            }
        )
    return matches


def _calls(source: str, callee: str) -> list[dict[str, str]]:
    tree = ast.parse(source)
    return [
        {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in node.keywords
            if keyword.arg
        }
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callee
    ]


@pytest.fixture
def small_df():
    return pd.DataFrame({
        "f1": list(range(200)),
        "f2": [x * 2 for x in range(200)],
        "target": [i % 2 for i in range(200)],
    })


def _call(df, base_dir, task_type="classification", target_col="target",
          holdout_fraction=0.2, random_seed=42):
    report_dir = base_dir / "rep"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = base_dir / "out" / "data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_outputs(
        df, _empty_report(), str(report_dir), str(out),
        delimiter=",",
        task_type=task_type,
        target_col=target_col,
        cfg={"holdout_fraction": holdout_fraction, "random_seed": random_seed},
    )
    return out.parent


def test_save_outputs_writes_train_and_holdout_siblings(tmp_path, small_df):
    out_dir = _call(small_df, tmp_path)
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "holdout.csv").exists()
    assert (out_dir / "holdout_manifest.json").exists()


def test_holdout_split_has_no_overlap(tmp_path, small_df):
    out_dir = _call(small_df, tmp_path)
    train = pd.read_csv(out_dir / "train.csv")
    holdout = pd.read_csv(out_dir / "holdout.csv")
    assert len(train) + len(holdout) == len(small_df)
    assert set(train["f1"]).isdisjoint(set(holdout["f1"]))


def test_holdout_split_is_deterministic(tmp_path, small_df):
    a_dir = tmp_path / "a"; a_dir.mkdir()
    b_dir = tmp_path / "b"; b_dir.mkdir()
    out_a = _call(small_df, a_dir)
    out_b = _call(small_df, b_dir)
    h1 = pd.read_csv(out_a / "holdout.csv").reset_index(drop=True)
    h2 = pd.read_csv(out_b / "holdout.csv").reset_index(drop=True)
    pd.testing.assert_frame_equal(h1, h2)


def test_holdout_split_classification_is_stratified(tmp_path):
    df = pd.DataFrame({
        "f1": list(range(200)),
        "target": [0] * 160 + [1] * 40,
    })
    out_dir = _call(df, tmp_path)
    holdout = pd.read_csv(out_dir / "holdout.csv")
    minority_in_holdout = (holdout["target"] == 1).sum()
    # 20% of 40 = 8; allow ±2 for rounding
    assert 6 <= minority_in_holdout <= 10, f"stratification broken: {minority_in_holdout}"


def test_stage3_scaler_fits_training_rows_only():
    df = pd.DataFrame(
        {
            "feature": [0.0, 2.0, 100.0],
            "target": [0, 1, 1],
            SPLIT_COLUMN: [TRAIN_PARTITION, TRAIN_PARTITION, HOLDOUT_PARTITION],
        }
    )

    transformed, _ = preprocess(
        df,
        "target",
        recipe_preprocessing={
            "encoding": {"categorical_method": "none"},
            "scaling": {"method": "standard"},
        },
        task_type="classification",
    )

    train_values = transformed.loc[
        transformed[SPLIT_COLUMN] == TRAIN_PARTITION, "feature"
    ]
    holdout_value = transformed.loc[
        transformed[SPLIT_COLUMN] == HOLDOUT_PARTITION, "feature"
    ].iloc[0]
    assert train_values.mean() == pytest.approx(0.0)
    assert holdout_value == pytest.approx(99.0)


def test_stage2_imputers_fit_training_rows_only():
    df = pd.DataFrame(
        {
            "numeric": [1.0, 3.0, None],
            "category": ["train-mode", "train-mode", None],
            "target": [0, 1, 1],
            SPLIT_COLUMN: [
                TRAIN_PARTITION,
                TRAIN_PARTITION,
                HOLDOUT_PARTITION,
            ],
        }
    )

    prepared, _, _ = prep_dataframe(
        df,
        "target",
        {"imputation_numeric": "median"},
        "classification",
        protected_columns=["target"],
    )

    holdout = prepared.loc[
        prepared[SPLIT_COLUMN] == HOLDOUT_PARTITION
    ].iloc[0]
    assert holdout["numeric"] == pytest.approx(2.0)
    assert holdout["category"] == "train-mode"


def test_stage2_cardinality_decision_uses_training_rows_only():
    df = pd.DataFrame(
        {
            "category": ["train", "train", "holdout-a", "holdout-b"],
            "target": [0, 1, 0, 1],
            SPLIT_COLUMN: [
                TRAIN_PARTITION,
                TRAIN_PARTITION,
                HOLDOUT_PARTITION,
                HOLDOUT_PARTITION,
            ],
        }
    )

    prepared, dropped, _ = prep_dataframe(
        df,
        "target",
        {"imputation_numeric": "median", "high_cardinality_max": 1},
        "classification",
        protected_columns=["target"],
    )

    assert dropped == []
    assert "category" in prepared.columns


def test_stage3_target_encoder_fits_training_rows_only(monkeypatch):
    observed = {}

    class _FakeTargetEncoder:
        def __init__(self, cols):
            self.cols = cols

        def fit(self, X, y):
            observed["fit_index"] = X.index.tolist()
            observed["target_index"] = y.index.tolist()
            return self

        def transform(self, X):
            return pd.DataFrame(
                {column: range(len(X)) for column in self.cols},
                index=X.index,
            )

    class _CategoryEncoders:
        TargetEncoder = _FakeTargetEncoder

    real_import = stage3_preprocessing.importlib.import_module

    def _import(name):
        if name == "category_encoders":
            return _CategoryEncoders
        return real_import(name)

    monkeypatch.setattr(stage3_preprocessing.importlib, "import_module", _import)
    df = pd.DataFrame(
        {
            "category": ["a", "b", "holdout-only"],
            "target": [0, 1, 1],
            SPLIT_COLUMN: [TRAIN_PARTITION, TRAIN_PARTITION, HOLDOUT_PARTITION],
        }
    )

    preprocess(
        df,
        "target",
        recipe_preprocessing={
            "encoding": {"categorical_method": "target"},
            "scaling": {"method": "none"},
        },
        task_type="classification",
    )

    assert observed["fit_index"] == [0, 1]
    assert observed["target_index"] == [0, 1]


def test_stage4_imputation_uses_training_median():
    train_count = 60
    df = pd.DataFrame(
        {
            "feature": [1.0] * 30 + [3.0] * 30 + [None],
            "target": [0, 1] * 30 + [1],
            SPLIT_COLUMN: [TRAIN_PARTITION] * train_count + [HOLDOUT_PARTITION],
        }
    )

    transformed, _, _, _ = feature_engineer(
        df,
        "target",
        "classification",
        {"stage4": {"apply_pca_threshold": 999}},
        {"method": "none"},
    )

    holdout_value = transformed.loc[
        transformed[SPLIT_COLUMN] == HOLDOUT_PARTITION, "feature"
    ].iloc[0]
    assert holdout_value == pytest.approx(2.0)


def test_stage4_variance_selection_ignores_holdout_variance():
    train_count = 100
    holdout_count = 20
    df = pd.DataFrame(
        {
            "signal": [i % 5 for i in range(train_count + holdout_count)],
            "holdout_only": [0.0] * train_count
            + [float(i) for i in range(holdout_count)],
            "target": [i % 2 for i in range(train_count + holdout_count)],
            SPLIT_COLUMN: [TRAIN_PARTITION] * train_count
            + [HOLDOUT_PARTITION] * holdout_count,
        }
    )

    transformed, kept, _, _ = feature_engineer(
        df,
        "target",
        "classification",
        {"stage4": {"apply_pca_threshold": 999}},
        {"method": "variance", "params": {"threshold": 0.01}},
    )

    assert kept == ["signal"]
    assert "holdout_only" not in transformed.columns


def test_stage4_pca_centers_on_training_rows_only():
    train_count = 100
    holdout_count = 20
    train_x1 = [float(i % 10) for i in range(train_count)]
    train_x2 = [float((i * 3) % 11) for i in range(train_count)]
    df = pd.DataFrame(
        {
            "x1": train_x1 + [1000.0 + i for i in range(holdout_count)],
            "x2": train_x2 + [2000.0 + i for i in range(holdout_count)],
            "target": [i % 2 for i in range(train_count + holdout_count)],
            SPLIT_COLUMN: [TRAIN_PARTITION] * train_count
            + [HOLDOUT_PARTITION] * holdout_count,
        }
    )

    transformed, _, pca_metadata, _ = feature_engineer(
        df,
        "target",
        "classification",
        {
            "stage4": {
                "apply_pca_threshold": 1,
                "pca_variance_retained": 0.95,
            }
        },
        {"method": "none"},
    )

    pc_columns = [column for column in transformed if column.startswith("PC")]
    train_pcs = transformed.loc[
        transformed[SPLIT_COLUMN] == TRAIN_PARTITION,
        pc_columns,
    ]
    assert pca_metadata["applied"] is True
    assert train_pcs.mean().abs().max() < 1e-10


def test_save_outputs_honors_preassigned_partition(tmp_path):
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            SPLIT_COLUMN: [
                TRAIN_PARTITION,
                HOLDOUT_PARTITION,
                TRAIN_PARTITION,
                HOLDOUT_PARTITION,
            ],
        }
    )

    out_dir = _call(df, tmp_path)
    train = pd.read_csv(out_dir / "train.csv")
    holdout = pd.read_csv(out_dir / "holdout.csv")
    combined = pd.read_csv(out_dir / "data.csv")

    assert train["feature"].tolist() == [1, 3]
    assert holdout["feature"].tolist() == [2, 4]
    assert SPLIT_COLUMN not in train.columns
    assert SPLIT_COLUMN not in holdout.columns
    assert SPLIT_COLUMN not in combined.columns
    assert ROW_ID_COLUMN not in train.columns
    assert ROW_ID_COLUMN in holdout.columns
    assert ROW_ID_COLUMN not in combined.columns
    assert holdout[ROW_ID_COLUMN].is_unique


def test_preassigned_partition_rejects_unassigned_rows():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            SPLIT_COLUMN: [TRAIN_PARTITION, HOLDOUT_PARTITION, None],
        }
    )

    with pytest.raises(ValueError, match="assign every row"):
        ensure_holdout_partition(
            df,
            target_col=None,
            task_type="clustering",
            holdout_fraction=0.2,
            random_seed=42,
        )


def test_chronological_partition_reserves_latest_rows():
    df = pd.DataFrame(
        {
            "event_time": [
                "2026-01-03",
                "2026-01-01",
                "2026-01-04",
                "2026-01-02",
                "2026-01-05",
            ],
            "feature": [3, 1, 4, 2, 5],
        }
    )

    partitioned = ensure_holdout_partition(
        df,
        target_col=None,
        task_type="regression",
        holdout_fraction=0.4,
        random_seed=42,
        split_strategy="chronological",
        time_column="event_time",
    )

    holdout_times = partitioned.loc[
        partitioned[SPLIT_COLUMN] == HOLDOUT_PARTITION,
        "event_time",
    ]
    assert set(holdout_times) == {"2026-01-04", "2026-01-05"}


def test_save_outputs_writes_declared_component_files(tmp_path):
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            SPLIT_COLUMN: [
                TRAIN_PARTITION,
                HOLDOUT_PARTITION,
                TRAIN_PARTITION,
                HOLDOUT_PARTITION,
            ],
        }
    )
    train_out = tmp_path / "declared" / "train.csv"
    holdout_out = tmp_path / "declared" / "holdout.csv"

    save_outputs(
        df,
        _empty_report(),
        str(tmp_path / "reports"),
        str(tmp_path / "combined.csv"),
        train_out=str(train_out),
        holdout_out=str(holdout_out),
        target_col="target",
    )

    assert pd.read_csv(train_out)["feature"].tolist() == [1, 3]
    assert pd.read_csv(holdout_out)["feature"].tolist() == [2, 4]


def test_azure_component_graph_declares_and_wires_holdout_boundary():
    stage2_component = yaml.safe_load(
        Path("components/stage2_preparation.yml").read_text(encoding="utf-8")
    )
    final_component = yaml.safe_load(
        Path("components/final_evaluation.yml").read_text(encoding="utf-8")
    )
    builder = Path("pipelines/pipeline_builder.py").read_text(encoding="utf-8")

    assert stage2_component["version"] == 13
    assert stage2_component["outputs"]["raw_train_out"]["type"] == "uri_file"
    assert stage2_component["outputs"]["raw_holdout_out"]["type"] == "uri_file"
    assert final_component["inputs"]["holdout_in"]["type"] == "uri_file"
    final_calls = _calls(builder, "final_eval")
    assert len(final_calls) == 2
    assert all(
        call["dataset_in"] == "s2.outputs.raw_train_out"
        and call["holdout_in"] == "s2.outputs.raw_holdout_out"
        and call["split_manifest_in"] == "s2.outputs.split_manifest_out"
        for call in final_calls
    )
    assert "holdout_in=s4.outputs.holdout_out" not in builder


def test_phaseb_mounts_only_stage2_raw_training_partition():
    source = Path("src/steps/s06_phaseb_variant_runner.py").read_text(
        encoding="utf-8"
    )
    builder = Path("pipelines/pipeline_builder.py").read_text(encoding="utf-8")
    stage2_component = yaml.safe_load(
        Path("components/stage2_preparation.yml").read_text(encoding="utf-8")
    )
    phaseb_component = yaml.safe_load(
        Path("components/s06_phaseb_variant_runner.yml").read_text(
            encoding="utf-8"
        )
    )

    assert stage2_component["outputs"]["raw_train_out"]["type"] == "uri_file"
    assert stage2_component["outputs"]["raw_holdout_out"]["type"] == "uri_file"
    phaseb_inputs = _dict_assignments(builder, "s06_kwargs")
    assert len(phaseb_inputs) == 2
    assert all(
        inputs["dataset_in"] == "s2.outputs.raw_train_out"
        and inputs["split_manifest"] == "s2.outputs.split_manifest_out"
        for inputs in phaseb_inputs
    )
    assert builder.count("holdout_in=s2.outputs.raw_holdout_out") == 2
    assert phaseb_component["inputs"]["split_manifest"]["type"] == "uri_file"
    runtime_source = source[source.index("def main():") :]
    assert "ensure_holdout_partition(" not in runtime_source
    assert "df_holdout" not in runtime_source


def test_stage2_raw_holdout_preserves_exact_locked_row_identity():
    source = pd.DataFrame(
        {
            "feature": list(range(12)),
            "category": ["a", "b"] * 6,
            "target": [0, 1] * 6,
        }
    )
    partitioned = ensure_holdout_partition(
        source,
        target_col="target",
        task_type="classification",
        holdout_fraction=0.25,
        random_seed=19,
        split_strategy="stratified",
    )
    expected_holdout = partitioned.loc[
        partitioned[SPLIT_COLUMN].eq(HOLDOUT_PARTITION)
    ]

    raw_train, raw_holdout = extract_raw_train_and_holdout(partitioned)

    assert len(raw_train) + len(raw_holdout) == len(source)
    assert ROW_ID_COLUMN not in raw_train
    assert raw_holdout[ROW_ID_COLUMN].tolist() == expected_holdout[
        ROW_ID_COLUMN
    ].tolist()
    assert raw_holdout[ROW_ID_COLUMN].is_unique
    assert raw_holdout.drop(columns=[ROW_ID_COLUMN]).reset_index(
        drop=True
    ).equals(
        expected_holdout.drop(
            columns=[SPLIT_COLUMN, ROW_ID_COLUMN]
        ).reset_index(drop=True)
    )


def test_stage1_recommendations_use_training_partition_only():
    source = Path("src/steps/stage1_ingestion.py").read_text(encoding="utf-8")

    assert "training_df = df.loc[df[SPLIT_COLUMN].eq(TRAIN_PARTITION)]" in source
    assert "generate_intelligent_recipe_recommendations(\n        training_df" in source
    assert "detect_time_series(training_df" in source


def test_final_evaluation_validation_fails_component(tmp_path):
    from steps.final_evaluation import enforce_input_validation

    args = SimpleNamespace(
        report_out=str(tmp_path / "report.json"),
        champion_out=str(tmp_path / "champion"),
    )
    validation = {
        "valid": False,
        "errors": ["Holdout dataset missing"],
        "warnings": [],
    }

    with pytest.raises(RuntimeError, match="final evaluation was not performed"):
        enforce_input_validation(args, validation)

    assert yaml.safe_load(Path(args.report_out).read_text())["status"] == "failed"


def test_final_evaluation_rejects_substituted_holdout_rows():
    from steps.final_evaluation import assert_matching_row_identity

    canonical = pd.Series(["row-a", "row-b"])
    substituted = pd.Series(["row-a", "row-c"])

    with pytest.raises(AssertionError):
        assert_matching_row_identity(substituted, canonical)
