# Installation

## Requirements

- Python 3.13 or higher
- Poetry (recommended) or pip

## Using Poetry (Recommended)

1. Clone the repository:

```bash
git clone <repository-url>
cd connect_4_rl
```

2. Install dependencies:

```bash
poetry install
```

3. Activate the virtual environment:

```bash
poetry shell
```

## Using pip

```bash
pip install -e .
```

## Dependencies

### Runtime Dependencies

| Package    | Version   | Purpose                          |
|------------|-----------|----------------------------------|
| gymnasium  | >= 1.0.0  | RL environment interface         |
| numpy      | >= 2.0.0  | Array operations                 |
| torch      | >= 2.0.0  | Neural network training          |
| pygame     | >= 2.6.0  | GUI for playing against AI       |

### Development Dependencies

| Package | Version  | Purpose         |
|---------|----------|-----------------|
| pytest  | >= 9.0.2 | Testing         |
| mypy    | >= 1.19  | Type checking   |
| pylint  | >= 4.0.4 | Code linting    |
| flake8  | >= 7.3.0 | Style checking  |

## Verifying Installation

Run the test suite to verify installation:

```bash
poetry run pytest tests/ -v
```

Run type checking:

```bash
poetry run mypy connect_4_rl/ --strict
```

Run linting:

```bash
poetry run pylint connect_4_rl/
poetry run flake8 connect_4_rl/
```

## Development Setup

For development, install with all dependencies:

```bash
poetry install --with dev
```

This includes testing and linting tools.
