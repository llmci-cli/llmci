# Scaffold

CI-native regression testing and migration for LLMs.

Catch LLM quality drops before they merge. Migrate models without breaking things.

## Installation

```bash
pip install scaffold-ai
```

## Quick Start

```bash
# Initialize a config
scaffold init

# Run evals
scaffold run

# Compare against main branch baseline
scaffold run --compare-to=main
```

## Documentation

See [PLAN.md](PLAN.md) for the full design document.

## License

Apache 2.0
