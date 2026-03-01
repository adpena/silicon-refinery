from fmtools import enhanced_debug


@enhanced_debug(prompt_to="crash_report_for_llm.txt")
def process_data(data_payload):
    """A buggy function that will inevitably crash."""
    print("Processing payload...")
    # This will trigger a TypeError because you can't add an integer to a string
    parsed_value = data_payload["value"] + 10
    return parsed_value


def main():
    print("🚀 Demonstrating FMTools's @enhanced_debug capabilities\n")

    bad_payload = {"value": "100"}

    try:
        process_data(bad_payload)
    except Exception:
        print(
            "\n✅ Execution continued gracefully in outer scope after FMTools dumped the analysis."
        )

    print("\nCheck your local directory for the 'crash_report_for_llm.txt' file!")


if __name__ == "__main__":
    main()
