from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data import (
    exploratory_data_analysis
)

def main() -> None:
    report = exploratory_data_analysis()
    print(f"- EDA directory: {report['eda_dir']}")
    print(f"- Charts: {report['charts']}")
    print(f"- Sample count: {report['sample_count']}")

if __name__ == "__main__":
    main()