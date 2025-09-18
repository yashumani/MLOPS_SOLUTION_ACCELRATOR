
"""
Data validation utilities.

Uses Pandera to validate schema and optionally Great Expectations for advanced checks.
"""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the schema of the DataFrame.

    Raises a ValueError if validation fails.

    Args:
        df (pd.DataFrame): Input data.
    """
    # Example schema: numeric columns must be floats or ints, others as strings
    columns = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            columns[col] = Column(pa.Float | pa.Int, nullable=True)
        else:
            columns[col] = Column(pa.String, nullable=True)

    schema = DataFrameSchema(columns)
    schema.validate(df, lazy=True)
