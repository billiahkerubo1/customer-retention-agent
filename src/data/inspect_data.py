
import pandas as pd

DATA_PATH = "data/raw/telco_customer_churn.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Shape: {df.shape}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\n========== COLUMN DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== BLANK STRING VALUES ==========")

    for column in df.select_dtypes(include=["str", "object"]).columns:
        blank_count = df[column].str.strip().eq("").sum()

        if blank_count > 0:
            print(f"{column}: {blank_count} blank values")

    print("\n========== TOTAL CHARGES INVESTIGATION ==========")

    total_charges = pd.to_numeric(
        df["TotalCharges"].str.strip(),
        errors="coerce"
    )

    invalid_values = total_charges.isna()

    print(f"Invalid numeric values: {invalid_values.sum()}")

    print("\nRows with invalid TotalCharges:")
    print(df.loc[invalid_values, ["customerID", "tenure", "TotalCharges"]])

    print("\n========== DUPLICATE CHECKS ==========")

    print(f"Duplicate rows: {df.duplicated().sum()}")

    print(
        "Duplicate customer IDs:",
        df["customerID"].duplicated().sum()
    )

    print("\n========== CATEGORICAL VALUE COUNTS ==========")

    categorical_columns = df.select_dtypes(
        include=["str", "object"]
    ).columns

    for column in categorical_columns:
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False))

    print("\n========== CHURN DISTRIBUTION ==========")

    churn_counts = df["Churn"].value_counts()

    churn_percentages = df["Churn"].value_counts(normalize=True) * 100

    print(churn_counts)
    print("\nPercentages:")
    print(churn_percentages.round(2))


if __name__ == "__main__":
    main()
