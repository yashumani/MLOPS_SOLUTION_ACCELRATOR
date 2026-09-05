FROM python:3.10.14-bookworm@sha256:9c0e621579faf384d982986f2e0ba86bf09619076842cd0fbd2f24a3bf09f0bc

SHELL ["/bin/bash", "-c"]

COPY requirements.lock /tmp/s12-registration-requirements.lock

RUN python -m pip install \
      --require-hashes \
      --requirement /tmp/s12-registration-requirements.lock \
    && python -m pip check \
    && python -c \
      "from azure.ai.ml import MLClient; import azureml.mlflow, catboost, lightgbm, marshmallow, xgboost, flaml; from flaml.automl.model import CatBoostEstimator, LGBMEstimator, LRL1Classifier, XGBoostLimitDepthEstimator; assert flaml.__version__ == '2.2.0'; assert marshmallow.__version__ == '3.23.3'"
