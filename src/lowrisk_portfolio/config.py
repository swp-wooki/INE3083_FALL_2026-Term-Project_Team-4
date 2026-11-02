"""Input coverage and estimation settings for the final-report experiment."""

from dataclasses import dataclass
from pathlib import Path

from .paths import DATA, OUTPUT


@dataclass(frozen=True)
class Config:
    data_dir: Path = DATA
    output_dir: Path = OUTPUT
    start: str = "2003-02"
    end: str = "2020-12"
    window: int = 36

    def validate(self):
        import pandas as pd

        if pd.Period(self.start, "M") > pd.Period(self.end, "M"):
            raise ValueError("Require start <= end")
        if self.window < 2:
            raise ValueError("Estimation window must contain at least two observations")
