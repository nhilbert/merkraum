# Contributing to Merkraum

Merkraum is an auditable knowledge memory layer for AI agents. Contributions are welcome.

## Getting Started

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Run tests: `python -m pytest test_unit.py test_security_review.py -v`
5. Open a pull request against `main`

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Review Process

Pull requests go through a two-layer review:

1. **CI checks** — automated linting (ruff), unit tests, and type checking run on every PR
2. **AI review** — an independent AI model reviews the PR for security issues, API compatibility, and code quality
3. **Human review** — a maintainer reviews and approves

All three layers must pass before merge.

## Code Standards

- Python 3.10+ compatibility
- Type hints for public functions
- No credentials or secrets in code — use environment variables
- Security-sensitive changes (auth, JWT, ACL) require explicit maintainer approval

## What We Look For

- Bug fixes with regression tests
- Performance improvements with benchmarks
- Documentation improvements
- New backend adapters (follow the `BackendAdapter` interface)

## License

By contributing, you agree that your contributions will be licensed under BSL 1.1 (see LICENSE).
