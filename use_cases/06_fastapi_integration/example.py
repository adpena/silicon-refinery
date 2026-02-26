from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import apple_fm_sdk as fm
from silicon_refinery import local_extract
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="SiliconRefinery Data Extraction API",
    description="A hyper-secure, zero-cloud REST API powered by Apple Silicon Neural Engine.",
    version="0.0.1",
)


# 1. Define our API Request Schema
class ExtractionRequest(BaseModel):
    document_text: str


# 2. Define our Apple Foundation Model Schema
@fm.generable()
class ExtractedEntity:
    primary_topic: str = fm.guide(description="The main subject of the document.")
    sentiment_score: int = fm.guide(
        range=(1, 10), description="Sentiment from 1 (negative) to 10 (positive)."
    )
    entities: list[str] = fm.guide(
        description="List of proper nouns, organizations, or key entities."
    )


# 3. Create our Zero-Trust local extraction pipeline
@local_extract(schema=ExtractedEntity, debug_timing=True)
async def process_document(raw_text: str) -> ExtractedEntity:
    """Analyze the text to extract the primary topic, sentiment score, and key entities."""
    pass


@app.post("/api/v1/extract")
async def extract_data(request: ExtractionRequest):
    """
    Synchronously extracts structured data from a provided text document.
    Data is processed entirely on the local Apple Silicon Neural Engine.
    """
    try:
        # Call our SiliconRefinery decorated function
        result = await process_document(request.document_text)

        # Convert the generable class to a dict for the JSON response
        return {
            "status": "success",
            "data": {
                "primary_topic": result.primary_topic,
                "sentiment_score": result.sentiment_score,
                "entities": result.entities,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 Starting SiliconRefinery API on http://localhost:8000")
    print(
        "Try sending a POST request to http://localhost:8000/api/v1/extract with JSON body: {'document_text': '...'}"
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
