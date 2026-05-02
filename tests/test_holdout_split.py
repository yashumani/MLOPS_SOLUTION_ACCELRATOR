"""Test holdout split written by stage4_feature_engineering.save_outputs."""

import pandas as pd
import pytest

from steps.stage4_feature_engineering import save_outputs


def _empty_report():
    return {"imbalance_metadata": {}}


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
