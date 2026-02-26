import polars as pl
import apple_fm_sdk as fm
import asyncio
import json


@pl.api.register_expr_namespace("local_llm")
class LocalLLMExpr:
    """
    A Polars extension that allows running local, on-device LLM inference
    directly inside Polars expressions: `df.select(pl.col("text").local_llm.extract(MySchema))`
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def extract(self, schema, instructions="Extract and structure the text."):
        # We define a function to process batches of strings
        def process_batch(series: pl.Series) -> pl.Series:
            async def run(s):
                model = fm.SystemLanguageModel()
                session = fm.LanguageModelSession(model=model, instructions=instructions)
                results = []
                for val in s:
                    if val is not None:
                        # Extract structured data
                        res = await session.respond(str(val), generating=schema)
                        # Serialize to JSON string for Polars compatibility
                        # Assuming the schema has __dict__ or vars() from fm.generable()
                        res_dict = vars(res) if hasattr(res, "__dict__") else str(res)
                        results.append(json.dumps(res_dict))
                    else:
                        results.append(None)
                return results

            # Run the async loop synchronously to bridge Polars (Rust) with Asyncio
            res_list = asyncio.run(run(series))
            return pl.Series(res_list)

        # Apply our batch processing function
        return self._expr.map_batches(process_batch, return_dtype=pl.String)
