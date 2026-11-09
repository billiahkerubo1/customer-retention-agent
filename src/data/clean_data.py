
import pandas as pd

RAW_DATA_PATH = "data/raw/telco_customer_churn.csv"

PROCESSED_DATA_PATH = (
    "data/processed/telco_customer_churn_clean.csv"
)


def load_data():
    """Load the original customer dataset."""
    return pd.read_csv(RAW_DATA_PATH)


def clean_data(df):
    """Clean and validate the customer dataset."""

    df = df.copy()

    # Remove accidental whitespace from column names
    df.columns = df.columns.str.strip()

    # Remove leading and trailing whitespace from text columns
    text_columns = df.select_dtypes(include=["object", "str"]).columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    # Convert TotalCharges from text to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Investigate missing TotalCharges values
    missing_total_charges = df["TotalCharges"].isna()

    missing_count = missing_total_charges.sum()

    print(f"Missing TotalCharges before cleaning: {missing_count}")

    # Verify that missing TotalCharges belongs to zero-tenure customers
    if missing_total_charges.any():
        zero_tenure = df.loc[
            missing_total_charges, "tenure"
        ].eq(0)

        if not zero_tenure.all():
            raise ValueError(
                "Missing TotalCharges found for customers "
                "with non-zero tenure. Further investigation required."
            )

        # Zero accumulated charges for new customers
        df.loc[missing_total_charges, "TotalCharges"] = 0

    # Validate the cleaned dataset
    if df["TotalCharges"].isna().any():
        raise ValueError("TotalCharges still contains missing values.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    if df["customerID"].isna().any():
        raise ValueError("Customer IDs cannot be missing.")

    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Unexpected values found in Churn.")

    return df


def main():
    df = load_data()

    print(f"Original dataset shape: {df.shape}")

    cleaned_df = clean_data(df)

    print(f"Cleaned dataset shape: {cleaned_df.shape}")

    print("\nCleaned data types:")
    print(cleaned_df.dtypes)

    print("\nRemaining missing values:")
    print(cleaned_df.isnull().sum())

    print("\nTotalCharges summary:")
    print(cleaned_df["TotalCharges"].describe())

    cleaned_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(f"\nCleaned dataset saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
