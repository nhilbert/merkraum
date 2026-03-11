# Public Repository Security & Privacy Readiness

This checklist is for verifying that the repository is safe to publish publicly.

## Current review status

- ✅ No hard-coded API keys, private keys, or passwords were found in tracked source files.
- ✅ Runtime secrets are loaded from environment variables and/or AWS Secrets Manager.
- ✅ `.env` is ignored by git and only `.env.example` is tracked.
- ✅ Default credentials in docs/config are clearly local-development defaults.
- ✅ No direct personal phone numbers, private addresses, or personal account credentials were found.
- ⚠️ Operational docs can accidentally include maintainer-identifying details (local usernames, home paths, personal names). Keep those generic.

## Guardrails for future commits

1. **Never commit real secrets**
   - Keep API keys/tokens only in local `.env` or secret managers.
   - Use obvious placeholders in docs/examples (e.g. `<your-api-key>`).

2. **Keep examples non-sensitive**
   - Avoid real user data in tests and docs.
   - Avoid local machine identifiers like `/home/<real-user>/...`.

3. **Preserve `.gitignore` hygiene**
   - Ensure `.env`, key files (`*.pem`, `*.key`), and local artifacts are ignored.

4. **Review before release**
   - Run a pattern scan for common secret formats.
   - Run tests/lint/type checks/build to ensure no accidental debug changes are shipped.

## Suggested pre-release commands

```bash
# quick secret scan in tracked files
rg -n -i "(api[_-]?key|secret|token|password|private[_-]?key|-----BEGIN|aws_access_key_id|aws_secret_access_key|ghp_|github_pat_|sk-)"

# test/lint/build checks (adjust to your environment)
pytest -q
python -m py_compile *.py
python -m build
```
