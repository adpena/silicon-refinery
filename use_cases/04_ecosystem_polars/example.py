import polars as pl
import apple_fm_sdk as fm

# Importing the module registers the `local_llm` namespace onto Polars DataFrames


@fm.generable()
class Ticket:
    department: str = fm.guide(anyOf=["IT", "HR", "Sales", "Billing", "Other"])
    urgency: int = fm.guide(description="Scale 1 to 5, where 5 is critical")


def main():
    # Load our real-world dataset directly into Polars
    df = pl.read_csv("datasets/support_tickets.csv")

    print("Original DataFrame (First 2 rows):")
    print(df.head(2))

    print("\nExecuting Local Inference directly inside Polars expression...")
    print("This runs entirely on Apple Silicon, adding zero cloud latency.\n")

    # We create a new column `extracted_json` using our custom namespace
    # The map_batches function under the hood chunks the df and runs the LLM
    enriched_df = df.with_columns(
        extracted_json=pl.col("email_body").local_llm.extract(schema=Ticket)
    )

    print("Enriched DataFrame:")
    print(enriched_df.select(["ticket_id", "email_subject", "extracted_json"]))


if __name__ == "__main__":
    main()
