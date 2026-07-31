"""Run format, lint, and type-check in one command."""

import subprocess
import sys


def main() -> None:
    commands = (
        ["ruff", "format", "."],
        ["ruff", "check", "."],
        ["mypy", "."],
        ["pytest", "tests"],
    )
    for command in commands:
        print(f"+ {' '.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
