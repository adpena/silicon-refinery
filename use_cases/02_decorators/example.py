import asyncio
import csv
from silicon_refinery import local_extract
import apple_fm_sdk as fm


@fm.generable()
class MedicalRecord:
    patient_symptoms: list[str] = fm.guide(description="List of isolated symptoms mentioned")
    suggested_triage: str = fm.guide(anyOf=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    duration_days: int = fm.guide(
        description="Integer representing how many days the symptoms have lasted, if mentioned. 0 if not."
    )


@local_extract(schema=MedicalRecord)
async def parse_doctor_notes(raw_text: str) -> MedicalRecord:
    """
    Extract structured medical data from raw dictated notes.
    Ensure triage urgency is inferred correctly based on the severity of symptoms.
    """
    pass


async def main():
    print("Processing Real-World Medical Notes via @local_extract...\n")

    with open("datasets/medical_notes.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            note = row["raw_note"]
            print(f"Raw Note: {note}")

            # The decorator handles the underlying LLM inference magically
            record = await parse_doctor_notes(note)

            print(f"-> Extracted Triage:   {record.suggested_triage}")
            print(f"-> Extracted Symptoms: {record.patient_symptoms}")
            print(f"-> Duration (days):    {record.duration_days}\n")


if __name__ == "__main__":
    asyncio.run(main())
