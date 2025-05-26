import pandas as pd
# import pandera as pa # You can uncomment these if you re-introduce specific schemas
# from pandera import Column, DataFrameSchema, Check

# -----------------------------------------------------------------------------
# Example Pandera Schema (Specific to college.csv - Keep for reference or future use)
# -----------------------------------------------------------------------------
# college_schema = DataFrameSchema({
#     "Private": Column(object, nullable=True), # Made nullable to be more general
#     "Apps": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "Accept": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "Enroll": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "Top10perc": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "Top25perc": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "F.Undergrad": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "P.Undergrad": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "Outstate": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "Room.Board": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "Books": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "Personal": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "PhD": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "Terminal": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "S.F.Ratio": Column(float, Check.ge(0), coerce=True, nullable=True),
#     "perc.alumni": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "Expend": Column(int, Check.ge(0), coerce=True, nullable=True),
#     "Grad.Rate": Column(int, Check.ge(0), coerce=True, nullable=True), # This was the original target
# }, strict=False, ordered=False) # Use strict=False to allow other columns not defined

# titanic_schema = DataFrameSchema({ ... you could define one for Titanic here ...})
# -----------------------------------------------------------------------------

def ingest_dataframe(path: str) -> pd.DataFrame:
    """
    Loads a DataFrame from a CSV file.
    Currently, detailed Pandera schema validation is bypassed for flexibility
    with different datasets. For production, a robust schema validation strategy
    per dataset or a dynamic schema approach is recommended.
    """
    print(f"INFO: data_ingest.py - Loading data from: {path}")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: data_ingest.py - File not found at {path}")
        raise
    except Exception as e:
        print(f"ERROR: data_ingest.py - Could not read CSV from {path}. Error: {e}")
        raise

    # To use a specific schema (e.g., if you had logic to choose one):
    # if path.endswith("college.csv"):
    #     print("INFO: data_ingest.py - Attempting to validate with college_schema.")
    #     try:
    #         df = college_schema.validate(df)
    #         print("INFO: data_ingest.py - college_schema validation successful.")
    #     except pa.errors.SchemaErrors as e:
    #         print("WARNING: data_ingest.py - Pandera schema validation failed for college.csv. Errors:")
    #         print(e.failure_cases) # Shows which rows/columns failed
    #         print("Proceeding with unvalidated DataFrame for college.csv.")
    # elif path.endswith("titanic.csv"):
    #     # df = titanic_schema.validate(df) # If you define titanic_schema
    #     print("INFO: data_ingest.py - No specific schema applied for titanic.csv yet.")
    # else:
    #     print("INFO: data_ingest.py - No specific Pandera schema applied for this dataset.")
    
    print(f"INFO: data_ingest.py - Successfully loaded DataFrame. Shape: {df.shape}")
    return df