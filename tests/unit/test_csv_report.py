from __future__ import annotations

import pandas as pd

from src.reporting.csv_report import write_csv
from src.schemas import ImageResult


def test_csv_preserves_awb_as_exact_string(tmp_path) -> None:
    value = "12345678901234567890"
    result = ImageResult(
        image_id="image-1",
        original_filename="parcel.jpg",
        awb_number=value,
        requires_review=False,
    )
    path = write_csv([result], tmp_path / "results.csv")
    dataframe = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert dataframe.loc[0, "awb_number"] == value
    assert dataframe.loc[0, "awb_number_excel"] == f'="{value}"'

