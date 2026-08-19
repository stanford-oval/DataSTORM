import datetime
import os
import time
from decimal import Decimal
from typing import Dict, Literal

from aiocache import caches, cached
from suql import suql_execute
from suql.postgresql_connection import execute_sql_with_column_info, apply_auto_limit

from knowledge_storm.datatalk_agent.suql_logger import log_suql_execution

# SUQL hardcodes its row-verification / field-classification model as the
# module constant `_VERIFICATION_MODEL_NAME = "gpt-5.2"`; the `llm_model_name`
# passed to suql_execute() does not reach it. Our genie proxy serves gpt-5.2
# only as "gpt-5.2-codex", so the default 400s ("no healthy deployments for
# this model"), which surfaces as an opaque `table "temp_table_..." does not
# exist` and silently returns zero rows for every answer()-in-WHERE query.
# gpt-5.4 is on the proxy and is in SUQL's reasoning_effort="none" list, which
# the tight 30-token verification budget needs — gpt-5/gpt-5-nano only reach
# "minimal" and can burn the budget on hidden reasoning, rejecting every row.
import suql.sql_free_text_support.execute_free_text_sql as _suql_free_text_sql

_suql_free_text_sql._VERIFICATION_MODEL_NAME = "gpt-5.4"


# All SUQL LLM traffic (compiler rewrites + per-row verification) bills to a
# dedicated lower-budget key, so a runaway SUQL query cannot drain the budget
# the rest of the pipeline depends on. The free-text server on :8510 is pointed
# at the same key by its launcher, so every SUQL call shares one budget.
# Falls back to the main key when the smaller one is not configured.
SUQL_API_KEY = os.getenv("DATASTORM_SMALLER_LITELLM_API_KEY") or os.getenv(
    "DATASTORM_LITELLM_API_KEY"
)
SUQL_API_BASE = os.getenv(
    "DATASTORM_SUQL_API_BASE", "https://api.openai.com/v1"
)

# litellm resolves `litellm_proxy/<model>` through these env vars. Setting them
# here rather than globally keeps the main pipeline — which passes its key
# explicitly to ChatOpenAI/OpenAIModel — on the main budget.
if SUQL_API_KEY:
    os.environ.setdefault("LITELLM_PROXY_API_KEY", SUQL_API_KEY)
    os.environ.setdefault("LITELLM_PROXY_API_BASE", SUQL_API_BASE)


# Configure the default cache to use Redis
caches.set_config({
    'default': {
        'cache': "aiocache.RedisCache",
        'endpoint': "localhost",
        'port': 6379,
        'serializer': {
            'class': "aiocache.serializers.JsonSerializer"
        },
        'plugins': []
    }
})

def prepare_initialize(table_schema):
    import pandas as pd
    # table_schema is passed in as a CSV file
    if type(table_schema) is str and os.path.exists(table_schema):
        with open(table_schema, 'r') as file:
            if file.read().strip():
                df = pd.read_csv(table_schema)
                table_schema_lst = df
                table_w_ids = {}
                for _, row in df.iterrows():
                    table_name = row['table_name']
                    table_w_ids[table_name] = row['id_field_name']
                return table_w_ids, table_schema_lst
            else:
                return {}, []
    else:
        raise ValueError(f"Cannot find schema file at path: {table_schema}")

def convert_sql_result_to_dict(results, column_names):
    """
    Convert SQL query results into a list of dictionaries keyed by column names.

    Each input row is expected to be an indexable sequence (e.g., tuple) whose
    positions align with `column_names`. Values are normalized for JSON-like
    consumption:
      - Decimal -> float
      - datetime.date -> 'YYYY-MM-DD' string
      - all other types are left unchanged

    Args:
        results: Iterable of rows from a DB driver (e.g., list[tuple]). If falsy
            (e.g., [] or None), it is returned as-is.
        column_names: Sequence of column names aligned with the row positions.

    Returns:
        list[dict]: One dictionary per row mapping column names to normalized values.

    Example:
        >>> convert_sql_result_to_dict([(1, Decimal('2.5'))], ['id', 'amount'])
        [{'id': 1, 'amount': 2.5}]
    """
    if not results:
        return results
    
    data = []
    for row in results:
        row_data = {}
        for col_index, col_value in enumerate(row):
            if isinstance(col_value, Decimal):
                col_value = float(col_value)
            if isinstance(col_value, datetime.date):
                col_value = col_value.strftime('%Y-%m-%d')
            row_data[column_names[col_index]] = col_value
        data.append(row_data)

    return data

@cached(ttl=3600)
async def execute_sql(
    sql: str,
    table_w_ids: Dict = {},
    database_name: str = "ingestion_experimental",
    suql_model_name="gpt-4-turbo",
    embedding_server_address: str = "http://127.0.0.1:8505",
    free_text_server_address: str = "http://127.0.0.1:8510",
    source_file_mapping: Dict = {},
    db_type: Literal["postgres", "sqlite"] = "postgres",
    db_secrets_file = None,
    suql_enabled = False,
    limit_query = " LIMIT 1000",
    db_path = None,
):
    if db_type == "postgres":
        try:
            if suql_enabled:
                sql = apply_auto_limit(sql, limit_query= " LIMIT 10")
                # Log the outcome of every SUQL query so failed/timed-out
                # ones can be replayed against an improved compiler. The
                # logger is a no-op unless the engine has set a path.
                _t0 = time.monotonic()
                _suql_err = None
                try:
                    results, column_names, _ = suql_execute(
                        sql,
                        table_w_ids,
                        database_name,
                        embedding_server_address=embedding_server_address,
                        free_text_server_address=free_text_server_address,
                        source_file_mapping=source_file_mapping,
                        llm_model_name=suql_model_name,
                        # Bill compiler + verification calls to the dedicated
                        # SUQL key rather than the pipeline's main budget.
                        api_key=SUQL_API_KEY,
                        api_base=SUQL_API_BASE,
                        disable_try_catch=True,
                        disable_try_catch_all_sql=True,
                        # SUQL queries call out to a free-text-answer server
                        # per row, which is slow on large tables; match the
                        # acled_suql_test.py default of 5 minutes.
                        statement_timeout=300000,
                    )
                except Exception as _e:
                    _suql_err = str(_e)
                    raise
                finally:
                    _dur_ms = (time.monotonic() - _t0) * 1000
                    if _suql_err is None:
                        _status = "ok"
                    elif "statement timeout" in _suql_err.lower() or "querycanceled" in _suql_err.lower():
                        _status = "timeout"
                    else:
                        _status = "error"
                    log_suql_execution(
                        sql=sql,
                        status=_status,
                        error=_suql_err,
                        duration_ms=_dur_ms,
                        database=database_name,
                    )
            else:
                results, column_names = execute_sql_with_column_info(
                    apply_auto_limit(sql, limit_query=limit_query),
                    database=database_name,
                    unprotected=True
                )
                column_names = list(map(lambda x:x[0], column_names))
            status = None
            results = convert_sql_result_to_dict(
                results, column_names
            )
        except Exception as e:
            results = None
            column_names = None
            status = str(e)
            print(f"error executing PostgreSQL query: {status}")

    elif db_type == "sqlite":
        try:
            import sqlite3

            # Apply limit to the query if specified
            if limit_query:
                sql = apply_auto_limit(sql, limit_query=limit_query)

            # Create a connection to the SQLite database
            conn = sqlite3.connect(db_path)

            # Enable column name access in results
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Execute the query
            cursor.execute(sql)

            # Get column names from cursor description
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch results
            results = [dict(row) for row in cursor.fetchall()]
            status = None

            # Close connection
            cursor.close()
            conn.close()

        except sqlite3.Error as e:
            results = None
            column_names = None
            status = f"SQLite error: {str(e)}"
            print(f"Error executing SQLite query: {status}")
        except Exception as e:
            results = None
            column_names = None
            status = str(e)
            print(f"Error with SQLite database: {status}")

    else:
        raise ValueError(f"Unsupported db_type: {db_type!r}")

    return results, status