import io

import httpx
import pandas as pd
from langchain_core.tools import tool


@tool
async def fetch_csv_as_dataframe(url: str) -> str:
    """
    Fetches a CSV file from a given public URL, loads it into a pandas DataFrame, 
    and returns a summary containing the columns, shape, and the first 5 rows.
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        summary = {
            "columns": list(df.columns),
            "shape": df.shape,
            "head": df.head(5).to_dict(orient="records"),
        }
        df.to_pickle(f"/tmp/{abs(hash(url))}.pkl")
        return str(summary)
    except Exception as e:
        # Return the error string to the agent so it knows the fetch failed!
        return f"Error fetching dataset from {url}: {str(e)}"


@tool
async def run_pandas_query(url: str, pandas_expression: str) -> str:
    """
    Executes a pandas python expression on a previously downloaded DataFrame.
    The DataFrame is available as the variable `df`.
    Example pandas_expression: 'df.groupby("state")["mortality"].max()'
    """
    path = f"/tmp/{abs(hash(url))}.pkl"
    df = pd.read_pickle(path)
    result = eval(pandas_expression)
    return str(result)


TOOLS = [fetch_csv_as_dataframe, run_pandas_query]
