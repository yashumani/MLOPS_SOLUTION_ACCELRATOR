# Runtime V33 Dependency Security Review

Date: 2026-09-02
Environment: `azureml:mlops-v3-unified:33`
Definition: `config/mlops_v3_unified_environment/conda_v33.yml`
LF-normalized definition SHA-256: `ba797e516569e9ae1141ab33b572660be13c93678f3272ba429b49d1970f6304`

## Decision

Runtime v33 is a release candidate, not yet release-qualified. Its 35 direct pip
dependencies resolve together. `pip-audit 2.10.1` reports seven records that
collapse to five distinct advisories in five packages. Each advisory below has
a source-backed non-applicability condition for this Azure ML batch runtime.
CI ignores only these exact advisory IDs and fails on any new advisory.

Azure ML registration, the environment smoke job, and end-to-end task
diagnostics remain required before this exception review can be accepted.

## Reviewed Exceptions

| Advisory | Package | Runtime applicability | Reopen condition |
|---|---|---|---|
| `PYSEC-2026-3447` | `setuptools==80.9.0` | Not applicable. The issue affects macOS source-distribution file exclusion. This Linux runtime installs wheels/conda packages and does not build or publish sdists. Setuptools remains pinned because newer versions removed the `pkg_resources` behavior required by `azureml-dataprep`. | Any sdist build/publish path, macOS build path, or AzureML dataprep version that supports fixed Setuptools. |
| `PYSEC-2024-110` | `scikit-learn==1.4.2` | Scanner range mismatch. The advisory text identifies versions through `1.4.1.post1`; v33 pins `1.4.2`. The repository does not use `TfidfVectorizer`. | Any TF-IDF feature path or a dependency scanner that identifies `1.4.2` with corrected upstream evidence. |
| `CVE-2026-71211` | `mlflow==3.15.0` | Not applicable to Azure-managed tracking client use. The vulnerable path is the self-hosted MLflow AI Gateway secret/proxy API. This repository does not launch an MLflow server, Gateway, or raw proxy. | Self-hosted MLflow server/Gateway deployment or exposure of its server handlers. |
| `PYSEC-2026-3552` | `cryptography==49.0.0` | Not applicable. Exploitation requires adaptive high-volume decryption of attacker-supplied PKCS#7 `EnvelopedData` through `pkcs7_decrypt_*`. The repository has no PKCS#7 decryption path. MLflow `3.15.0` requires `cryptography<50`, while the current Azure MLflow package caps `mlflow-skinny<=3.15.0`. | Any PKCS#7/S-MIME decryption path, untrusted envelope processing, or Azure MLflow release that permits MLflow/cryptography versions containing the fix. |
| `PYSEC-2026-3740` | `nltk==3.10.3` | Not applicable. NLTK is transitive through Evidently; the repository does not call the affected parser, perceptron, tagger, or maxent model path APIs and does not rely on NLTK pathsec containment. | Any NLTK model import/export or caller-controlled model path. |

## Source Checks

The repository-wide search for the following execution paths returned no
matches outside documentation and dependency metadata:

- `pkcs7_decrypt_*`
- `TransitionParser`, `AveragedPerceptron`, `PerceptronTagger`, `save_maxent_params`
- `TfidfVectorizer`
- `mlflow server`, `_create_gateway_secret`, `gateway_api`, `raw_proxy`
- `setup.py sdist`, `python -m build`

## Required Verification

1. CI must resolve the v33 direct pins, pass `pip check`, pass the allowlisted
   runtime audit, and return a clean API dependency audit.
2. The Azure smoke job must import the runtime modules, compare every direct
   installed version with `conda_v33.yml`, run Evidently, and write an MLflow
   tracking run to the configured Azure workspace.
3. The exact Azure image digest and full environment lock must be captured.
4. Classification, regression, and clustering diagnostics must pass using v33.
5. Re-run the audit and this applicability review before final production
   approval if any direct version or listed reopen condition changes.
