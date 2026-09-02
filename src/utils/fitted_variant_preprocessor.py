"""Serializable fitted preprocessing graph for Phase B raw-input bundles."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


SUPPORTED_RECIPE_METHODS = {
    "imputation": frozenset(
        {
            "mean",
            "median",
            "knn",
            "iterative",
            "mode",
            "most_frequent",
            "constant",
            "zero_fill",
            "trimmed_mean",
            "winsorized_mean",
            "numeric_mean_cat_mode",
            "numeric_median_cat_mode",
        }
    ),
    "encoding": frozenset({"none", "label", "onehot"}),
    "scaling": frozenset(
        {"none", "standard", "robust", "minmax", "yeo_johnson", "quantile"}
    ),
    "imbalance": frozenset(
        {"none", "smote", "adasyn", "smoteenn", "smotetomek"}
    ),
    "outlier": frozenset({"none", "iqr_capping", "winsorize"}),
    "feature_selection": frozenset(
        {"none", "correlation", "variance", "mutual_info"}
    ),
}


def _materialize_owned_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Detach transformed columns from read-only or memory-mapped buffers."""

    data = {
        column: np.array(frame[column].to_numpy(copy=False), copy=True)
        for column in frame.columns
    }
    return pd.DataFrame(data, index=frame.index.copy(), copy=True)


class FittedVariantPreprocessor(BaseEstimator, TransformerMixin):
    """Fit recipe transforms once on training data and replay them at inference."""

    def __init__(self, recipe: Mapping[str, Any], *, random_seed: int = 42):
        self.recipe = recipe
        self.random_seed = random_seed

    def _stage3(self) -> dict[str, Any]:
        return dict(self.recipe.get("stage3_preprocessing") or {})

    def _stage4(self) -> dict[str, Any]:
        return dict(self.recipe.get("stage4_feature_engineering") or {})

    def fit(self, frame: pd.DataFrame, target: Any = None):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("FittedVariantPreprocessor requires non-empty DataFrame")
        self.input_columns_ = [str(column) for column in frame.columns]
        self.numeric_columns_ = frame.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns_ = [
            column for column in frame.columns if column not in self.numeric_columns_
        ]
        transformed = frame.copy()
        stage3 = self._stage3()
        imputation = dict(stage3.get("imputation") or {})
        method = str(imputation.get("method", "mean"))
        self.numeric_imputer_ = None
        self.numeric_fill_values_: dict[str, Any] = {}
        if self.numeric_columns_:
            numeric = transformed[self.numeric_columns_]
            if method == "knn":
                from sklearn.impute import KNNImputer

                self.numeric_imputer_ = KNNImputer(
                    n_neighbors=int(imputation.get("n_neighbors") or 5),
                    weights="distance",
                ).fit(numeric)
            elif method == "iterative":
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer

                self.numeric_imputer_ = IterativeImputer(
                    random_state=self.random_seed,
                    max_iter=int(imputation.get("max_iter") or 10),
                ).fit(numeric)
            elif method in {"median", "numeric_median_cat_mode"}:
                self.numeric_fill_values_ = numeric.median().to_dict()
            elif method in {"mode", "most_frequent"}:
                self.numeric_fill_values_ = {
                    column: (
                        numeric[column].mode().iloc[0]
                        if not numeric[column].mode().empty
                        else 0
                    )
                    for column in self.numeric_columns_
                }
            elif method == "constant":
                value = imputation.get("fill_value", 0)
                self.numeric_fill_values_ = {
                    column: value for column in self.numeric_columns_
                }
            elif method == "zero_fill":
                self.numeric_fill_values_ = {
                    column: 0 for column in self.numeric_columns_
                }
            elif method == "trimmed_mean":
                from scipy.stats import trim_mean

                fraction = float(imputation.get("trim_fraction") or 0.1)
                self.numeric_fill_values_ = {
                    column: float(
                        trim_mean(numeric[column].dropna().values, fraction)
                    )
                    for column in self.numeric_columns_
                }
            elif method == "winsorized_mean":
                from scipy.stats.mstats import winsorize

                fraction = float(imputation.get("trim_fraction") or 0.05)
                self.numeric_fill_values_ = {
                    column: float(
                        winsorize(
                            numeric[column].dropna().values,
                            limits=[fraction, fraction],
                        ).mean()
                    )
                    for column in self.numeric_columns_
                }
            else:
                self.numeric_fill_values_ = numeric.mean().to_dict()
        self.categorical_fill_values_ = {
            column: (
                transformed[column].mode().iloc[0]
                if not transformed[column].mode().empty
                else "missing"
            )
            for column in self.categorical_columns_
        }
        transformed = self._apply_imputation(transformed)

        outlier = dict(stage3.get("outlier_handling") or {})
        outlier_method = str(outlier.get("method", "none"))
        self.outlier_bounds_: dict[str, tuple[float, float]] = {}
        numeric_after = transformed.select_dtypes(include=[np.number]).columns.tolist()
        if outlier_method == "iqr_capping":
            for column in numeric_after:
                q1 = float(transformed[column].quantile(0.25))
                q3 = float(transformed[column].quantile(0.75))
                iqr = q3 - q1
                self.outlier_bounds_[column] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        elif outlier_method == "winsorize":
            for column in numeric_after:
                self.outlier_bounds_[column] = (
                    float(transformed[column].quantile(0.05)),
                    float(transformed[column].quantile(0.95)),
                )
        transformed = self._apply_outlier_bounds(transformed)

        encoding = dict(stage3.get("encoding") or {})
        self.encoding_method_ = str(encoding.get("categorical_method", "onehot"))
        self.category_values_: dict[str, list[Any]] = {
            column: sorted(
                transformed[column].dropna().unique().tolist(),
                key=lambda value: str(value),
            )
            for column in self.categorical_columns_
        }
        transformed = self._apply_encoding(transformed)
        self.encoded_columns_ = [str(column) for column in transformed.columns]

        scaling = dict(stage3.get("scaling") or {})
        scaling_method = str(scaling.get("method", "none"))
        self.scaled_columns_ = transformed.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.scaler_ = None
        if scaling_method != "none" and self.scaled_columns_:
            if scaling_method == "standard":
                from sklearn.preprocessing import StandardScaler

                self.scaler_ = StandardScaler()
            elif scaling_method == "robust":
                from sklearn.preprocessing import RobustScaler

                self.scaler_ = RobustScaler()
            elif scaling_method == "minmax":
                from sklearn.preprocessing import MinMaxScaler

                self.scaler_ = MinMaxScaler()
            elif scaling_method == "yeo_johnson":
                from sklearn.preprocessing import PowerTransformer

                self.scaler_ = PowerTransformer(
                    method="yeo-johnson", standardize=True
                )
            elif scaling_method == "quantile":
                from sklearn.preprocessing import QuantileTransformer

                self.scaler_ = QuantileTransformer(
                    output_distribution="normal",
                    random_state=self.random_seed,
                    n_quantiles=min(1000, len(transformed)),
                )
            else:
                raise ValueError(f"Unsupported scaling method: {scaling_method!r}")
            self.scaler_.fit(transformed[self.scaled_columns_])
        transformed = self._apply_scaling(transformed)

        feature_selection = dict(
            self._stage4().get("feature_selection") or {}
        )
        selection_method = str(feature_selection.get("method", "none"))
        configured_threshold = feature_selection.get("threshold", 0.01)
        threshold = float(
            0.01 if configured_threshold is None else configured_threshold
        )
        self.selected_columns_ = [str(column) for column in transformed.columns]
        if selection_method != "none":
            numeric_columns = transformed.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            non_numeric = [
                column for column in transformed.columns if column not in numeric_columns
            ]
            if selection_method == "variance":
                from sklearn.feature_selection import VarianceThreshold

                selector = VarianceThreshold(threshold=max(0.0, threshold))
                selector.fit(transformed[numeric_columns])
                selected = [
                    column
                    for column, keep in zip(
                        numeric_columns, selector.get_support()
                    )
                    if keep
                ]
            else:
                if target is None:
                    raise ValueError(
                        f"Feature selection method {selection_method!r} "
                        "requires a target"
                    )
                target_series = pd.Series(target).reset_index(drop=True)
                if not pd.api.types.is_numeric_dtype(target_series):
                    target_series = pd.Series(
                        pd.Categorical(target_series).codes,
                        index=target_series.index,
                    )
            if selection_method == "correlation":
                scores = transformed[numeric_columns].reset_index(
                    drop=True
                ).corrwith(target_series).abs()
                selected = scores[scores >= threshold].index.tolist()
            elif selection_method == "mutual_info":
                from sklearn.feature_selection import (
                    mutual_info_classif,
                    mutual_info_regression,
                )

                recipe_task_type = str(self.recipe.get("task_type") or "").lower()
                if recipe_task_type == "classification":
                    scorer = mutual_info_classif
                elif recipe_task_type == "regression":
                    scorer = mutual_info_regression
                else:
                    raise ValueError(
                        "Mutual-information feature selection requires recipe "
                        "task_type 'classification' or 'regression'"
                    )
                values = scorer(
                    transformed[numeric_columns].fillna(0),
                    target_series,
                    random_state=self.random_seed,
                )
                selected = [
                    column
                    for column, score in zip(numeric_columns, values)
                    if float(score) >= threshold
                ]
            elif selection_method != "variance":
                raise ValueError(
                    f"Unsupported feature selection method: {selection_method!r}"
                )
            ordered = [
                column
                for column in transformed.columns
                if column in set(selected + non_numeric)
            ]
            if not ordered:
                raise ValueError(
                    f"Feature selection method {selection_method!r} removed "
                    "all features"
                )
            self.selected_columns_ = ordered
        return self

    def _apply_imputation(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if self.numeric_columns_:
            if self.numeric_imputer_ is not None:
                result[self.numeric_columns_] = self.numeric_imputer_.transform(
                    result[self.numeric_columns_]
                )
            else:
                result[self.numeric_columns_] = result[
                    self.numeric_columns_
                ].fillna(self.numeric_fill_values_)
        for column, value in self.categorical_fill_values_.items():
            result[column] = result[column].fillna(value)
        return result

    def _apply_outlier_bounds(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column, (lower, upper) in self.outlier_bounds_.items():
            result[column] = result[column].clip(lower=lower, upper=upper)
        return result

    def _apply_encoding(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if self.encoding_method_ == "label":
            for column, values in self.category_values_.items():
                mapping = {value: index for index, value in enumerate(values)}
                result[column] = result[column].map(mapping).fillna(-1).astype(int)
        elif self.encoding_method_ == "onehot":
            for column, values in self.category_values_.items():
                result[column] = result[column].astype(
                    pd.CategoricalDtype(categories=values)
                )
            result = pd.get_dummies(
                result,
                columns=list(self.category_values_),
                drop_first=True,
            )
            if hasattr(self, "encoded_columns_"):
                result = result.reindex(columns=self.encoded_columns_, fill_value=0)
        elif self.encoding_method_ != "none":
            raise ValueError(
                f"Unsupported inference-safe encoding: {self.encoding_method_!r}"
            )
        return result

    def _apply_scaling(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if self.scaler_ is not None:
            result[self.scaled_columns_] = self.scaler_.transform(
                result[self.scaled_columns_]
            )
        return result

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "input_columns_"):
            raise ValueError("FittedVariantPreprocessor is not fitted")
        missing = [column for column in self.input_columns_ if column not in frame]
        if missing:
            raise ValueError(f"Raw input is missing columns: {missing}")
        result = frame.loc[:, self.input_columns_].copy()
        result = self._apply_imputation(result)
        result = self._apply_outlier_bounds(result)
        result = self._apply_encoding(result)
        result = self._apply_scaling(result)
        ordered = result.reindex(columns=self.selected_columns_, fill_value=0)
        return _materialize_owned_frame(ordered)

    def fit_transform(self, frame: pd.DataFrame, target: Any = None) -> pd.DataFrame:
        return self.fit(frame, target).transform(frame)
