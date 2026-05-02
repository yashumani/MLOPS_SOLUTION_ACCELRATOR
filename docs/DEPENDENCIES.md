# Dependency Reproducibility - V3

## Current Files

V3 currently keeps two dependency source files in the repository:

| File | Purpose | Audience |
|---|---|---|
| `requirements.txt` | Runtime dependencies used by the Azure ML environment build. | Azure ML environment build and operators. |
| `requirements.in` | Maintainer-edited dependency source with bounded requirements and rationale. | Maintainers. |

`requirements.lock` is not currently committed. When the team is ready for a hash-pinned local or CI install, generate it from `requirements.in` and validate it before committing.

## Update Workflow

Edit `requirements.in`, then regenerate a lock file when a locked workflow is required:

```bash
pip install pip-tools
pip-compile --generate-hashes --resolver=backtracking \
  --output-file=requirements.lock requirements.in
```

Install from a generated lock file:

```bash
pip install --require-hashes -r requirements.lock
```

## SBOM

The repository includes `scripts/generate_sbom.sh` for CycloneDX SBOM generation.

```bash
bash scripts/generate_sbom.sh
```

Expected output:

```text
sbom/sbom-cyclonedx.json
```

## Audit

When a lock file exists, audit it with:

```bash
pip install pip-audit
pip-audit -r requirements.lock --format json --output audit.json
```

## Production Note

Dependency updates should be treated as production changes. Do not update package bounds in the same commit as step-script fixes unless the runtime failure is dependency-related.
