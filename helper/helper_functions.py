from pathlib import Path
from typing import Any
import json
import pandas as pd

from app.config import settings

def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
