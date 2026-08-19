import asyncio
import hashlib
import json
import math
import os
import pickle
import re
import time
import warnings
from collections import defaultdict
from typing import List

import redis.asyncio as redis
import requests
from json_repair import repair_json
from litellm import get_max_tokens, token_counter
from pydantic import BaseModel

from knowledge_storm.datatalk_agent.state import SqlQuery
from knowledge_storm.langfuse_llm import call_llm_with_structured_output, get_llm
from knowledge_storm.log_utils import logger


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)  # from ReFinED


class BaseParser:
    @classmethod
    def initialize(engine: str):
        raise NotImplementedError("Subclasses should implement this method")

    @classmethod
    def run_batch(cls, questions: list[str]):
        return asyncio.run(
            cls.runnable.with_config(
                {"recursion_limit": 60, "max_concurrency": 50}
            ).abatch(questions)
        )


async def parse_string_to_json(output: str) -> dict:
    return repair_json(output, return_objects=True)


def extract_code_block_from_output(llm_output: str, code_block: str) -> str:
    code_block = code_block.lower()
    if f"```{code_block}" in llm_output.lower():
        start_idx = llm_output.lower().rfind(f"```{code_block}") + len(
            f"```{code_block}"
        )
        end_idx = llm_output.lower().rfind("```", start_idx)
        if end_idx < 0:
            # The LLM may not emit the closing fence; fall through to end of output.
            end_idx = len(llm_output)
        extracted_block = llm_output[start_idx:end_idx].strip()
        return extracted_block
    else:
        raise ValueError(f"Expected a code block, but llm output is {llm_output}")


def format_table_schema(schema: str) -> str:
    # TODO: Used to format the table schemas from sql, right now we just return the schema as is.
    return schema


def extract_psql_comments(sql):
    # Split the SQL into lines
    lines = sql.strip().split('\n')

    comments = []
    sql_without_comments = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('--'):
            # Collect comments
            comment = stripped_line.lstrip('- ').rstrip()
            comments.append(comment)
        else:
            sql_without_comments.append(line)

    # Reconstruct the SQL without comments
    sql_no_comments = '\n'.join(sql_without_comments)

    return '\n'.join(comments)


def get_tables(schema) -> dict:
    non_enum_schemas = schema[schema.apply(lambda x: x["type"] != "enum", axis=1)]
    res = non_enum_schemas[["table_name", "table_CREATE_command"]].to_dict(orient='records')
    for i in res:
        i["comments"] = extract_psql_comments(i["table_CREATE_command"])
        del i["table_CREATE_command"]
    
    return res

def retrieve_tables_details(utterance: List[str], schema) -> dict:
    res = schema[schema.apply(lambda x:x["table_name"] in utterance, axis=1)]
    
    return res["table_CREATE_command"].astype(str).tolist()

# Takes in entity table name and a list of entities that the embedding search found
# Retrieves their associated descripriotn sfrom the entity table and returns as dictionary
async def llm_select_entities_in_batches_async(
    search_str,
    all_distinct_values,
    entity_table_name,
    fill_prompt=True,
    shard_size=52,
    num_shuffled_iters=1,
    voting_threshold=1,
    engine="gpt-4o-mini",
    langfuse_readonly: bool = False,
):
    """Stub: the live entity-linking pipeline used to read
    ``llm_config.yaml`` (chainlite-era prompt-dirs config) and a local
    ``select_entities.prompt`` file from disk to size shards before
    calling ``call_llm_with_structured_output("datatalk_select_entities",
    ...)``. We dropped the chainlite/llm_config path entirely — Langfuse
    is the single source of truth for prompts now — but the
    ``entity_linking`` action that consumed this function is also disabled
    in every domain's ``available_actions``, so this code path was already
    unreachable in the live pipeline.

    If you re-enable ``entity_linking`` in the future, replace this body
    with a Langfuse-fetched prompt-text equivalent (use
    ``compile_prompt('datatalk_select_entities', ...)`` to retrieve the
    raw text for token counting / sharding).
    """
    raise NotImplementedError(
        "entity_linking is disabled; llm_select_entities_in_batches_async "
        "previously read llm_config.yaml + select_entities.prompt from "
        "disk, both of which were removed when the agent moved to Langfuse-only "
        "prompt management. Re-implement on top of compile_prompt() before "
        "re-enabling the entity_linking action."
    )

class RedisLLMCacheEntityLinking:
    _instance = None
    _locks = {}  # Class-level locks shared across all instances
    _redis_client = None
    _redis_url = None
    _ttl = None
    _last_connection_time = 0
    _connection_timeout = 300  # 5 minutes connection timeout

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RedisLLMCacheEntityLinking, cls).__new__(cls)
        return cls._instance

    def __init__(self, redis_url="redis://localhost:6379", ttl=3600):
        # Store configuration but don't create the connection yet
        RedisLLMCacheEntityLinking._redis_url = redis_url
        RedisLLMCacheEntityLinking._ttl = ttl

    async def _get_redis_client(self):
        """Get or create Redis client with connection management"""
        current_time = time.time()
        
        # Create a new connection if:
        # 1. No connection exists
        # 2. Connection is too old (timeout exceeded)
        if (RedisLLMCacheEntityLinking._redis_client is None or 
            current_time - RedisLLMCacheEntityLinking._last_connection_time > RedisLLMCacheEntityLinking._connection_timeout):
            
            # Close any existing connection
            await self._close_redis_client()
            
            # Create a new connection
            try:
                RedisLLMCacheEntityLinking._redis_client = redis.from_url(
                    RedisLLMCacheEntityLinking._redis_url, 
                    decode_responses=False,  # Binary mode for pickled objects
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    health_check_interval=30
                )
                RedisLLMCacheEntityLinking._last_connection_time = current_time
                logger.info("Created new Redis connection for entity linking cache")
            except Exception as e:
                logger.error(f"Error creating Redis connection for entity linking: {str(e)}")
                return None
                
        return RedisLLMCacheEntityLinking._redis_client
        
    async def _close_redis_client(self):
        """Close the Redis client if it exists"""
        if RedisLLMCacheEntityLinking._redis_client is not None:
            try:
                await RedisLLMCacheEntityLinking._redis_client.close()
                logger.info("Closed Redis connection for entity linking cache")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {str(e)}")
            finally:
                RedisLLMCacheEntityLinking._redis_client = None

    @classmethod
    async def _hash_key(cls, input_text):
        """Generate a unique key for caching based on input text"""
        return hashlib.sha256(input_text.encode()).hexdigest()

    async def entity_linking_caching(
        self, 
        search_str,
        entity_table_nm,
        db_secrets_file,
        location_cols = ['stanford_api_data.location', 'stanford_api_data.country', 'stanford_api_data.region', 'stanford_api_data.iso'],
        free_text_cols = ['stanford_api_data.notes'],
        db_type = 'mysql',
        langfuse_readonly: bool = False
    ):
        """Cached LLM call with Redis"""
        input_text = f"{search_str}-{entity_table_nm}-{db_secrets_file}-{','.join(location_cols)}-{','.join(free_text_cols)}-{db_type}"

        key = await self._hash_key(input_text)
        
        # Get Redis client
        redis_client = await self._get_redis_client()
        if redis_client is None:
            # If Redis is unavailable, fall back to uncached operation
            return await entity_linking(
                search_str=search_str,
                entity_table_nm=entity_table_nm,
                db_secrets_file=db_secrets_file,
                location_cols=location_cols,
                free_text_cols=free_text_cols,
                db_type=db_type,
                langfuse_readonly=langfuse_readonly
            )
            
        # Try to get from cache
        try:
            cached_result = await redis_client.get(key)
            if cached_result:
                return pickle.loads(cached_result)  # Deserialize and return cached response
        except Exception as e:
            logger.warning(f"Redis get operation failed: {str(e)}")
            # Continue with uncached operation if Redis fails

        # Use lock to prevent duplicate work
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        async with self._locks[key]:  # Ensures only one concurrent request per unique input
            try:
                # Double-check cache
                cached_result = await redis_client.get(key)
                if cached_result:
                    return pickle.loads(cached_result)
                    
                # Get actual result
                result = await entity_linking(
                    search_str=search_str,
                    entity_table_nm=entity_table_nm,
                    db_secrets_file=db_secrets_file,
                    location_cols=location_cols,
                    free_text_cols=free_text_cols,
                    db_type=db_type,
                    langfuse_readonly=langfuse_readonly
                )

                # Store the result in cache
                try:
                    await redis_client.setex(key, RedisLLMCacheEntityLinking._ttl, pickle.dumps(result))
                except Exception as e:
                    logger.warning(f"Redis set operation failed: {str(e)}")
                
                return result
            finally:
                # Always clean up the lock
                if key in self._locks:
                    del self._locks[key]

    @classmethod
    async def close(cls):
        """Close Redis connection - should be called during application shutdown"""
        if cls._instance is not None:
            await cls._instance._close_redis_client()
            cls._locks = {}  # Clear all locks
            logger.info("RedisLLMCacheEntityLinking connections and resources cleaned up")

    @classmethod
    async def close(cls):
        """Close the underlying Redis client and reset the singleton instance."""
        try:
            if cls._instance is not None and hasattr(cls._instance, "redis") and cls._instance.redis:
                try:
                    # Prefer aclose if available (redis.asyncio >= 5)
                    if hasattr(cls._instance.redis, "aclose") and callable(getattr(cls._instance.redis, "aclose")):
                        await cls._instance.redis.aclose()
                    else:
                        await cls._instance.redis.close()

                    # Disconnect connection pool (supports both async and sync)
                    pool = getattr(cls._instance.redis, "connection_pool", None)
                    if pool is not None:
                        disconnect = getattr(pool, "disconnect", None)
                        if disconnect is not None:
                            maybe_coro = disconnect()
                            if asyncio.iscoroutine(maybe_coro):
                                await maybe_coro
                except Exception as e:
                    logger.warning(f"Error closing Redis client in RedisLLMCacheEntityLinking: {e}")
        finally:
            # Always reset internal state so a fresh instance can be created later
            cls._instance = None
            cls._locks.clear()

async def entity_linking(
    search_str,
    entity_table_nm,
    db_secrets_file,
    location_cols = ['stanford_api_data.location', 'stanford_api_data.country', 'stanford_api_data.region', 'stanford_api_data.iso'],
    free_text_cols = ['stanford_api_data.notes'],
    db_type = 'mysql',
    langfuse_readonly: bool = False
):
    if set(location_cols) & set(entity_table_nm):
        return f"'I should not use entity_linking on a location-related column. Instead, I should use location_linking action to resolve locations.'"
    elif set(free_text_cols) & set(entity_table_nm):
        return f"'I should not use entity_linking on free-text columns. I should instead try to filter on different columns, only use free-text columns in projection, or use a LIKE operator on free-text column if neccesarry."  # possibly add option to allow %LIKE% operators in this ase
    
    output_entities = []


    # async def get_all_entities(table, column):
    #     # TODO: consider caching this for each column since it takes some time
    #     distinct_query = f"SELECT DISTINCT {col_name} FROM {table_name};"
    #     result_dicts, _ = await execute_sql(distinct_query, db_type=db_type, limit_query=None, db_secrets_file=db_secrets_file)
    #     distinct_values = [dic[col_name] for dic in result_dicts]
    #     return distinct_values

    # For now, we are using Genie search API that includes descriptions for a subset of entities
    # This will be swapped out with a different endpoint to retrieve from all ACLED entities
    async def get_top_n_entities_from_embedding(
        search_str,
        n = 1000,
        retriever_endpoint: str = os.getenv("DATATALK_ENTITY_RETRIEVER_ENDPOINT", "")
    ):

        payload = {
            'query': search_str,
            'num_blocks': n,
            'rerank': False,
            'num_blocks_to_rerank': n
        }

        print(f'payload is: {payload}')

        response = requests.post(retriever_endpoint, json=payload)

        if response.status_code == 429:
            raise Exception("Rate limit reached. Try again later.")
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        results = response.json()
        entity_dicts = results[0]['results']

        filtered_entities = [dic['document_title'] for dic in entity_dicts]
        return filtered_entities

    if entity_table_nm == 'actors':
        distinct_values = await get_top_n_entities_from_embedding(search_str, n=100)

        # print(f'Retrieved {len(distinct_values)} entities from embedding retrieval for the search string: {search_str} to do filtering on in entity_linking')

        # do filtration with LLM selection
        ents = await llm_select_entities_in_batches_async(
            search_str,
            distinct_values,
            entity_table_nm,
            langfuse_readonly=langfuse_readonly
        )

        # print(f'Narrowed it down to {len(ents)} entities using LLM classification')

        output_entities = list(ents)

    else:
        return "Currently, entity_linking only supports ACLED dataset actors"

    return output_entities

async def postprocess_entities(query, substitution_dict):
    comment = ""
    if not substitution_dict:
        return query

    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, substitution_dict.keys())) + r')\b')
    revised_query = pattern.sub(lambda m: substitution_dict[m.group()], query)
    
    if revised_query != query:
        comment = "-- The agent's original query contained variable names to represent entities, which were expanded into the full list in post-processing."

    return comment + '\n' + revised_query

async def location_linking(location_lst, langfuse_readonly: bool = False):
    input_error_msg = "I encountered an error in resolving locations. I should try again, ensuring the input arguments adhere to requirements."

    async def get_location_conversion_info(location_options_dict):
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # LLM resolves location
                get_location_chain = call_llm_with_structured_output(
                    "datatalk_select_locations",
                    variables={
                        "location_list": location_lst,
                        "database_type": 'mysql',
                        "location_options_dict": location_options_dict
                    }, # TODO needs to be updated
                    output_class=LocationConversionInfo,
                    llm=get_llm(model_name="gpt-4o"),
                    langfuse_readonly=langfuse_readonly,
                )
                
                # Parse json response
                try:
                    output = repair_json(output, return_objects=True)
                except Exception as json_error:
                    raise ValueError(f"Failed to convert LLM response into dict: {str(json_error)}")
                
                # Validate it is a dict
                if not isinstance(output, dict):
                    raise ValueError("LLM output is not a valid dictionary")
                
                # Validate all locations are present (if it missed one, retry)
                if set(output.keys()) != set(location_lst):
                    raise ValueError("The location_linker did not link locations for every location search string")
                
                break
            
            except Exception:
                if attempt == max_retries - 1:
                    output = input_error_msg
        
        return output


    def get_location_from_azure(query):
        EARTH_RADIUS = 6371000  # meters
        TOLERANCE = 1500  # meters

        subscription_key = os.environ["AZURE_MAP_KEY"]
        # API endpoint
        url = "https://atlas.microsoft.com/search/address/json"

        # Parameters for the request
        params = {
            "subscription-key": subscription_key,
            "api-version": "1.0",
            "language": "en-US",
            "query": query,
        }
        # Sending the GET request
        response = requests.get(url, params=params)

        if response.status_code != 200:
            return "Azure API request failed with status code {response.status_code}: {response.text}", 0

        # Extracting the JSON response
        response_json = response.json()

        if "results" not in response_json or not response_json["results"] or "type" not in response_json["results"][0] or 'boundingBox' not in response_json["results"][0]:
            return "The Azure API returned no results for the given location - I should try again with a different search string.",0

        if response_json["results"][0]["type"] == "Geography":
            bbox = response_json["results"][0]["boundingBox"]
            latitude_north, longitude_west = (
                bbox["topLeftPoint"]["lat"],
                bbox["topLeftPoint"]["lon"],
            )
            latitude_south, longitude_east = (
                bbox["btmRightPoint"]["lat"],
                bbox["btmRightPoint"]["lon"],
            )
        else:
            # get coords
            coord = response_json["results"][0]["position"]
            longitude, latitude = coord["lon"], coord["lat"]

            # Get location range
            delta_longitude = TOLERANCE / EARTH_RADIUS * 180 / math.pi
            delta_latitude = (
                TOLERANCE
                / (EARTH_RADIUS * math.cos(latitude / 180 * math.pi))
                * 180
                / math.pi
            )

            longitude_west = longitude - delta_longitude
            longitude_east = longitude + delta_longitude
            latitude_south = latitude - delta_latitude
            latitude_north = latitude + delta_latitude

        return ((longitude_west, longitude_east, latitude_south, latitude_north), 1)
    
    with open('location_options_dict.json', 'r') as file:
        location_options_dict = json.load(file)
    output = await get_location_conversion_info(location_options_dict)

    if output == input_error_msg:
        return input_error_msg

    # Else format responses with right coordinate substitution
    for loc in output:
        if not output[loc]:  # LLM deemed search string needing of lat/long coords
            coord_output = get_location_from_azure(loc)
            
            if not coord_output[1]:
                output[loc] = coord_output[0]
            else:
                longitude_west, longitude_east, latitude_south, latitude_north = coord_output[0]
                output[loc] = f"longitude BETWEEN {longitude_west} AND {longitude_east} AND latitude BETWEEN {latitude_south} AND {latitude_north}"
    
    return output
