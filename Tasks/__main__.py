import sys
from pathlib import Path

from .loader import validate_dataset

target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "dataset"
problems = validate_dataset(target)

if not problems:
    print(f"OK: all tasks in {target} are valid")
else:
    for name, errors in problems.items():
        print(f"{name}:")
        for e in errors:
            print(f"  - {e}")
    sys.exit(1)
