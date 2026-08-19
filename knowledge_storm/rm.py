import asyncio
import csv
import hashlib
import json
import logging
import os
import pathlib
import threading
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import backoff
import dspy
import requests
from dsp import backoff_hdlr, giveup_hdlr

from .utils import WebPageHelper


def _append_failed_query_to_repo_root(
    query: str,
    error: str,
    conversation_history: Any = None,
    rm_name: str = "unknown",
    debug: Any = None,
    filename: str = "failed_queries.jsonl",
) -> None:
    """
    Best-effort logging of failed queries to a file at the repo root (one level above `knowledge_storm/`).
    """
    try:
        repo_root = Path(__file__).resolve().parents[1]
        out_path = repo_root / filename
        payload = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "error": error,
            "conversation_history": conversation_history,
            "rm_name": rm_name,
            "debug": debug,
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Never allow logging failures to impact the main execution path.
        pass


def _truncate_for_log(value: Any, limit: int = 8000) -> Any:
    """
    Best-effort truncation for debug logging payloads to keep JSONL lines reasonably sized.
    - If value is a string, truncate it.
    - Otherwise, stringify then truncate.
    """
    try:
        if value is None:
            return None
        if isinstance(value, str):
            return value if len(value) <= limit else (value[:limit] + "...(truncated)")
        s = str(value)
        return s if len(s) <= limit else (s[:limit] + "...(truncated)")
    except Exception:
        return "<unserializable>"


def _get_table_summary_statistics_from_results_table(results_table):
    """
    Compute per-column summary statistics from a SQL results table (list[dict]).

    Mirrors the logic requested by the user:
    - distinct_percentage
    - min/max (+ median/mean when possible)
    - top 5 values (+ counts) when not all values are distinct
    """
    if results_table is None:
        return None
    if not results_table:  # Empty result ([], {}, "", etc.)
        return {}

    try:
        import pandas as pd  # lazy import to avoid hard dependency at import time
    except Exception:
        return None

    try:
        df = pd.DataFrame(results_table)
    except Exception:
        return None

    stats = {}
    for column in df.columns:
        col_stats = {}

        # Number of distinct values + percentage
        try:
            distinct_values = df[column].nunique()
            total_values = len(df)
            distinct_percentage = (distinct_values / total_values * 100) if total_values > 0 else 0
            col_stats["distinct_percentage"] = f"{round(distinct_percentage, 2)}%"
        except Exception:
            distinct_percentage = 100

        # Min/max (+ median/mean) for numeric/date/time-like columns
        try:
            sample_value = df[column].iloc[0] if not df[column].empty else None
            if sample_value is not None:
                if isinstance(sample_value, (int, float, pd.Timestamp, pd.Timedelta)) or pd.api.types.is_numeric_dtype(df[column]):
                    try:
                        col_stats["min"] = df[column].min()
                        col_stats["max"] = df[column].max()
                        try:
                            col_stats["median"] = df[column].median()
                            col_stats["mean"] = df[column].mean()
                        except Exception:
                            pass
                    except Exception:
                        pass
        except Exception:
            pass

        # Top 5 values and their counts (only if not all values are distinct)
        try:
            if distinct_percentage < 100:
                value_counts = df[column].value_counts().head(5).to_dict()
                col_stats["top_values"] = [{"value": k, "count": v} for k, v in value_counts.items()]
        except Exception:
            pass

        stats[column] = col_stats

    return stats


def _format_summary_stats_markdown(summary_stats: dict) -> str:
    if not summary_stats:
        return ""
    out = "## Summary Statistics\n\n"
    for column, stats in summary_stats.items():
        out += f"### {column}\n"
        for stat, value in (stats or {}).items():
            out += f"{stat}: {value}\n"
        out += "\n"
    return out


class YouRM(dspy.Retrieve):
    def __init__(self, ydc_api_key=None, k=3, is_valid_source: Callable = None):
        super().__init__(k=k)
        if not ydc_api_key and not os.environ.get("YDC_API_KEY"):
            raise RuntimeError(
                "You must supply ydc_api_key or set environment variable YDC_API_KEY"
            )
        elif ydc_api_key:
            self.ydc_api_key = ydc_api_key
        else:
            self.ydc_api_key = os.environ["YDC_API_KEY"]
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"YouRM": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with You.com for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        for query in queries:
            try:
                headers = {"X-API-Key": self.ydc_api_key}
                results = requests.get(
                    f"https://api.ydc-index.io/search?query={query}",
                    headers=headers,
                ).json()

                authoritative_results = []
                for r in results["hits"]:
                    if self.is_valid_source(r["url"]) and r["url"] not in exclude_urls:
                        authoritative_results.append(r)
                if "hits" in results:
                    collected_results.extend(authoritative_results[: self.k])
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")

        return collected_results


class BingSearch(dspy.Retrieve):
    def __init__(
        self,
        bing_search_api_key=None,
        k=3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
        mkt="en-US",
        language="en",
        **kwargs,
    ):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            mkt, language, **kwargs: Bing search API parameters.
            - Reference: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
        """
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("BING_SEARCH_API_KEY"):
            raise RuntimeError(
                "You must supply bing_search_subscription_key or set environment variable BING_SEARCH_API_KEY"
            )
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["BING_SEARCH_API_KEY"]
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.params = {"mkt": mkt, "setLang": language, "count": k, **kwargs}
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"BingSearch": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with Bing for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}

        for query in queries:
            try:
                results = requests.get(
                    self.endpoint, headers=headers, params={**self.params, "q": query}
                ).json()

                for d in results["webPages"]["value"]:
                    if self.is_valid_source(d["url"]) and d["url"] not in exclude_urls:
                        url_to_results[d["url"]] = {
                            "url": d["url"],
                            "title": d["name"],
                            "description": d["snippet"],
                        }
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(
            list(url_to_results.keys())
        )
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r["snippets"] = valid_url_to_snippets[url]["snippets"]
            collected_results.append(r)

        return collected_results


class SerperRM(dspy.Retrieve):
    """Retrieve information from custom queries using Serper.dev."""

    def __init__(
        self,
        serper_search_api_key=None,
        k=3,
        query_params=None,
        ENABLE_EXTRA_SNIPPET_EXTRACTION=False,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
    ):
        """Args:
        serper_search_api_key str: API key to run serper, can be found by creating an account on https://serper.dev/
        query_params (dict or list of dict): parameters in dictionary or list of dictionaries that has a max size of 100 that will be used to query.
            Commonly used fields are as follows (see more information in https://serper.dev/playground):
                q str: query that will be used with google search
                type str: type that will be used for browsing google. Types are search, images, video, maps, places, etc.
                gl str: Country that will be focused on for the search
                location str: Country where the search will originate from. All locates can be found here: https://api.serper.dev/locations.
                autocorrect bool: Enable autocorrect on the queries while searching, if query is misspelled, will be updated.
                results int: Max number of results per page.
                page int: Max number of pages per call.
                tbs str: date time range, automatically set to any time by default.
                qdr:h str: Date time range for the past hour.
                qdr:d str: Date time range for the past 24 hours.
                qdr:w str: Date time range for past week.
                qdr:m str: Date time range for past month.
                qdr:y str: Date time range for past year.
        """
        super().__init__(k=k)
        self.usage = 0
        self.query_params = None
        self.ENABLE_EXTRA_SNIPPET_EXTRACTION = ENABLE_EXTRA_SNIPPET_EXTRACTION
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )

        if query_params is None:
            self.query_params = {"num": k, "autocorrect": True, "page": 1}
        else:
            self.query_params = query_params
            self.query_params.update({"num": k})
        self.serper_search_api_key = serper_search_api_key
        if not self.serper_search_api_key and not os.environ.get("SERPER_API_KEY"):
            raise RuntimeError(
                "You must supply a serper_search_api_key param or set environment variable SERPER_API_KEY"
            )

        elif self.serper_search_api_key:
            self.serper_search_api_key = serper_search_api_key

        else:
            self.serper_search_api_key = os.environ["SERPER_API_KEY"]

        self.base_url = "https://google.serper.dev"

    def serper_runner(self, query_params):
        self.search_url = f"{self.base_url}/search"

        headers = {
            "X-API-KEY": self.serper_search_api_key,
            "Content-Type": "application/json",
        }

        response = requests.request(
            "POST", self.search_url, headers=headers, json=query_params
        )

        if response == None:
            raise RuntimeError(
                f"Error had occurred while running the search process.\n Error is {response.reason}, had failed with status code {response.status_code}"
            )

        return response.json()

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"SerperRM": usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str]):
        """
        Calls the API and searches for the query passed in.


        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): Dummy parameter to match the interface. Does not have any effect.

        Returns:
            a list of dictionaries, each dictionary has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )

        self.usage += len(queries)
        self.results = []
        collected_results = []
        for query in queries:
            if query == "Queries:":
                continue
            query_params = self.query_params

            # All available parameters can be found in the playground: https://serper.dev/playground
            # Sets the json value for query to be the query that is being parsed.
            query_params["q"] = query

            # Sets the type to be search, can be images, video, places, maps etc that Google provides.
            query_params["type"] = "search"

            self.result = self.serper_runner(query_params)
            self.results.append(self.result)

        # Array of dictionaries that will be used by Storm to create the jsons
        collected_results = []

        if self.ENABLE_EXTRA_SNIPPET_EXTRACTION:
            urls = []
            for result in self.results:
                organic_results = result.get("organic", [])
                for organic in organic_results:
                    url = organic.get("link")
                    if url:
                        urls.append(url)
            valid_url_to_snippets = self.webpage_helper.urls_to_snippets(urls)
        else:
            valid_url_to_snippets = {}

        for result in self.results:
            try:
                # An array of dictionaries that contains the snippets, title of the document and url that will be used.
                organic_results = result.get("organic")
                knowledge_graph = result.get("knowledgeGraph")
                for organic in organic_results:
                    snippets = [organic.get("snippet")]
                    if self.ENABLE_EXTRA_SNIPPET_EXTRACTION:
                        snippets.extend(
                            valid_url_to_snippets.get(url.strip("'"), {}).get(
                                "snippets", []
                            )
                        )
                    collected_results.append(
                        {
                            "snippets": snippets,
                            "title": organic.get("title"),
                            "url": organic.get("link"),
                            "description": (
                                knowledge_graph.get("description")
                                if knowledge_graph is not None
                                else ""
                            ),
                        }
                    )
            except:
                continue

        # Filter out results with URLs starting with "https://acleddata.com/"
        # NOTE: hardcode
        filtered_results = [result for result in collected_results if not result["url"].startswith("https://acleddata.com/")]
        return filtered_results


class BraveRM(dspy.Retrieve):
    def __init__(
        self, brave_search_api_key=None, k=3, is_valid_source: Callable = None
    ):
        super().__init__(k=k)
        if not brave_search_api_key and not os.environ.get("BRAVE_API_KEY"):
            raise RuntimeError(
                "You must supply brave_search_api_key or set environment variable BRAVE_API_KEY"
            )
        elif brave_search_api_key:
            self.brave_search_api_key = brave_search_api_key
        else:
            self.brave_search_api_key = os.environ["BRAVE_API_KEY"]
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"BraveRM": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with api.search.brave.com for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        for query in queries:
            try:
                headers = {
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.brave_search_api_key,
                }
                response = requests.get(
                    f"https://api.search.brave.com/res/v1/web/search?result_filter=web&q={query}",
                    headers=headers,
                ).json()
                results = response.get("web", {}).get("results", [])

                for result in results:
                    collected_results.append(
                        {
                            "snippets": result.get("extra_snippets", []),
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "description": result.get("description"),
                        }
                    )
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")

        return collected_results


class SearXNG(dspy.Retrieve):
    def __init__(
        self,
        searxng_api_url,
        searxng_api_key=None,
        k=3,
        is_valid_source: Callable = None,
    ):
        """Initialize the SearXNG search retriever.
        Please set up SearXNG according to https://docs.searxng.org/index.html.

        Args:
            searxng_api_url (str): The URL of the SearXNG API. Consult SearXNG documentation for details.
            searxng_api_key (str, optional): The API key for the SearXNG API. Defaults to None. Consult SearXNG documentation for details.
            k (int, optional): The number of top passages to retrieve. Defaults to 3.
            is_valid_source (Callable, optional): A function that takes a URL and returns a boolean indicating if the
            source is valid. Defaults to None.
        """
        super().__init__(k=k)
        if not searxng_api_url:
            raise RuntimeError("You must supply searxng_api_url")
        self.searxng_api_url = searxng_api_url
        self.searxng_api_key = searxng_api_key
        self.usage = 0

        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"SearXNG": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with SearxNG for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        headers = (
            {"Authorization": f"Bearer {self.searxng_api_key}"}
            if self.searxng_api_key
            else {}
        )

        for query in queries:
            try:
                params = {"q": query, "format": "json"}
                response = requests.get(
                    self.searxng_api_url, headers=headers, params=params
                )
                results = response.json()

                for r in results["results"]:
                    if self.is_valid_source(r["url"]) and r["url"] not in exclude_urls:
                        collected_results.append(
                            {
                                "description": r.get("content", ""),
                                "snippets": [r.get("content", "")],
                                "title": r.get("title", ""),
                                "url": r["url"],
                            }
                        )
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")

        return collected_results


class DuckDuckGoSearchRM(dspy.Retrieve):
    """Retrieve information from custom queries using DuckDuckGo."""

    def __init__(
        self,
        k: int = 3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
        safe_search: str = "On",
        region: str = "us-en",
    ):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            **kwargs: Additional parameters for the OpenAI API.
        """
        super().__init__(k=k)
        try:
            from duckduckgo_search import DDGS
        except ImportError as err:
            raise ImportError(
                "Duckduckgo requires `pip install duckduckgo_search`."
            ) from err
        self.k = k
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )
        self.usage = 0
        # All params for search can be found here:
        #   https://duckduckgo.com/duckduckgo-help-pages/settings/params/

        # Sets the backend to be api
        self.duck_duck_go_backend = "api"

        # Only gets safe search results
        self.duck_duck_go_safe_search = safe_search

        # Specifies the region that the search will use
        self.duck_duck_go_region = region

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

        # Import the duckduckgo search library found here: https://github.com/deedy5/duckduckgo_search
        self.ddgs = DDGS()

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"DuckDuckGoRM": usage}

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_time=1000,
        max_tries=8,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, query: str):
        results = self.ddgs.text(
            query, max_results=self.k, backend=self.duck_duck_go_backend
        )
        return results

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with DuckDuckGoSearch for self.k top passages for query or queries
        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.
        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        collected_results = []

        for query in queries:
            #  list of dicts that will be parsed to return
            results = self.request(query)

            for d in results:
                # assert d is dict
                if not isinstance(d, dict):
                    print(f"Invalid result: {d}\n")
                    continue

                try:
                    # ensure keys are present
                    url = d.get("href", None)
                    title = d.get("title", None)
                    description = d.get("description", title)
                    snippets = [d.get("body", None)]

                    # raise exception of missing key(s)
                    if not all([url, title, description, snippets]):
                        raise ValueError(f"Missing key(s) in result: {d}")
                    if self.is_valid_source(url) and url not in exclude_urls:
                        result = {
                            "url": url,
                            "title": title,
                            "description": description,
                            "snippets": snippets,
                        }
                        collected_results.append(result)
                    else:
                        print(f"invalid source {url} or url in exclude_urls")
                except Exception as e:
                    print(f"Error occurs when processing {result=}: {e}\n")
                    print(f"Error occurs when searching query {query}: {e}")

        return collected_results


class TavilySearchRM(dspy.Retrieve):
    """Retrieve information from custom queries using Tavily. Documentation and examples can be found at https://docs.tavily.com/docs/python-sdk/tavily-search/examples"""

    def __init__(
        self,
        tavily_search_api_key=None,
        k: int = 3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
        include_raw_content=False,
    ):
        """
        Params:
            tavily_search_api_key str: API key for tavily that can be retrieved from https://tavily.com/
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            include_raw_content bool: Boolean that is used to determine if the full text should be returned.
        """
        super().__init__(k=k)
        try:
            from tavily import TavilyClient
        except ImportError as err:
            raise ImportError("Tavily requires `pip install tavily-python`.") from err

        if not tavily_search_api_key and not os.environ.get("TAVILY_API_KEY"):
            raise RuntimeError(
                "You must supply tavily_search_api_key or set environment variable TAVILY_API_KEY"
            )
        elif tavily_search_api_key:
            self.tavily_search_api_key = tavily_search_api_key
        else:
            self.tavily_search_api_key = os.environ["TAVILY_API_KEY"]

        self.k = k
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )

        self.usage = 0

        # Creates client instance that will use search. Full search params are here:
        # https://docs.tavily.com/docs/python-sdk/tavily-search/examples
        self.tavily_client = TavilyClient(api_key=self.tavily_search_api_key)

        self.include_raw_content = include_raw_content

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"TavilySearchRM": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with TavilySearch for self.k top passages for query or queries
        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.
        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        collected_results = []

        for query in queries:
            args = {
                "max_results": self.k,
                "include_raw_contents": self.include_raw_content,
            }
            #  list of dicts that will be parsed to return
            responseData = self.tavily_client.search(query)
            results = responseData.get("results")
            for d in results:
                # assert d is dict
                if not isinstance(d, dict):
                    print(f"Invalid result: {d}\n")
                    continue

                try:
                    # ensure keys are present
                    url = d.get("url", None)
                    title = d.get("title", None)
                    description = d.get("content", None)
                    snippets = []
                    if d.get("raw_body_content"):
                        snippets.append(d.get("raw_body_content"))
                    else:
                        snippets.append(d.get("content"))

                    # raise exception of missing key(s)
                    if not all([url, title, description, snippets]):
                        raise ValueError(f"Missing key(s) in result: {d}")
                    if self.is_valid_source(url) and url not in exclude_urls:
                        result = {
                            "url": url,
                            "title": title,
                            "description": description,
                            "snippets": snippets,
                        }
                        collected_results.append(result)
                    else:
                        print(f"invalid source {url} or url in exclude_urls")
                except Exception as e:
                    print(f"Error occurs when processing {result=}: {e}\n")
                    print(f"Error occurs when searching query {query}: {e}")

        return collected_results


# Create a custom logger
logger = logging.getLogger(__name__)

# Set the overall log level
logger.setLevel(logging.ERROR)

# Create handlers
file_handler = logging.FileHandler('request_errors.log')
# console_handler = logging.StreamHandler()

# Set log level for handlers
file_handler.setLevel(logging.WARNING)
# console_handler.setLevel(logging.WARNING)

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
# logger.addHandler(console_handler)
logger.propagate = False

# Dedicated debug logger for DatatalkRM I/O
debug_log_path = os.getenv(
    'DATASTORM_DEBUG_LOG',
    str(pathlib.Path(__file__).resolve().parent.parent / 'log' / 'debug_rm.log'),
)
try:
    os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
except Exception:
    # If directory creation fails, fall back to current directory
    debug_log_path = 'debug_rm.log'

rm_debug_logger = logging.getLogger('rm_debug')
rm_debug_logger.setLevel(logging.DEBUG)

# Avoid adding duplicate handlers if module is reloaded
_existing_debug_paths = [
    getattr(h, 'baseFilename', None) for h in rm_debug_logger.handlers if isinstance(h, logging.FileHandler)
]
if os.path.abspath(debug_log_path) not in [os.path.abspath(p) for p in _existing_debug_paths if p]:
    _debug_file_handler = logging.FileHandler(debug_log_path)
    _debug_file_handler.setLevel(logging.DEBUG)
    _debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    _debug_file_handler.setFormatter(_debug_formatter)
    rm_debug_logger.addHandler(_debug_file_handler)

rm_debug_logger.propagate = False

def upload_to_azure(source):
    import subprocess
    import os
    Azure_SAS_token = os.environ.get("AZURE_SAS_TOKEN")
    Azure_storage_dest = os.environ.get("AZURE_STORAGE_DEST")
    if not Azure_SAS_token:
        raise RuntimeError("AZURE_SAS_TOKEN environment variable is not set.")
    if not Azure_storage_dest:
        raise RuntimeError("AZURE_STORAGE_DEST environment variable is not set.")
    
    filename = source.split('/')[-1]
    destination = f'{Azure_storage_dest}/{filename}?{Azure_SAS_token}'

    command = [
        'azcopy', 'copy', source, destination
    ]
    # Execute the command and capture output. If it fails, let it raise so callers
    # get a full stack trace (instead of silently falling back).
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return f"{Azure_storage_dest}/{filename}"


class DatatalkRM(dspy.Retrieve):
    """In-process Datatalk retrieval module.

    Calls the vendored ``knowledge_storm.datatalk_agent`` library directly
    rather than POSTing to the FastAPI endpoint that used to live in
    datatalk_domains. The library is async; this class wraps it so the
    sync ``forward()`` interface (called from TopicExpert via Retriever's
    threadpool) still works.
    """

    SQL_RESULTS_DIR = os.getenv(
        "DATASTORM_SQL_RESULTS_DIR",
        str(pathlib.Path(__file__).resolve().parent.parent / "sql_results"),
    )

    def __init__(self, k=3, is_valid_source: Callable = None, domain="acled", disable_upload_to_azure: bool = False, enable_python: bool = False, langfuse_readonly: bool = False, include_summary_stats: bool = True, engine: str = None):
        self.usage = 0
        self.domain = domain
        self.rm_type = "datatalk"
        self.disable_upload_to_azure = disable_upload_to_azure
        self.enable_python = enable_python
        self.langfuse_readonly = langfuse_readonly
        self.include_summary_stats = include_summary_stats
        self.engine = engine

        # The DatatalkParser holds DB connections + table schema. Build it
        # lazily on the first forward() call (init is async + does file I/O
        # against $DATATALK_DOMAINS_ROOT/<domain>/...). Threads racing on
        # the first call are coordinated by `_parser_init_lock`; subsequent
        # calls share the parser.
        self._parser = None
        self._parser_init_lock = threading.Lock()
        # The parser's LangGraph runnable holds per-call state via the
        # streamed events. We serialize concurrent run_single_message
        # calls through one parser instance to avoid surprising
        # interleavings; the tree-search fan-out is small enough that
        # this isn't a meaningful throughput hit.
        self._call_lock = threading.Lock()

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"DatatalkRM": usage}

    def _ensure_parser(self, *, langfuse_readonly: bool, engine: str):
        """Build the DatatalkParser on first use; cache it on the instance."""
        if self._parser is not None:
            return self._parser
        with self._parser_init_lock:
            if self._parser is not None:
                return self._parser
            from knowledge_storm.datatalk_agent import library  # local import: keeps import cost off cold paths
            self._parser = asyncio.run(
                library.on_chat_start(
                    domain=self.domain,
                    enable_chainlit=False,
                    engine=engine or "gpt-5",
                    enable_python=self.enable_python,
                    langfuse_readonly=langfuse_readonly,
                )
            )
            return self._parser

    def _build_url(self, response: dict, query: str, *, disable_upload_to_azure: bool) -> str:
        """Mirror the legacy URL behavior: optional Azure upload of the saved CSV.

        - When upload is disabled or the library didn't write a CSV, fall
          back to a deterministic ``upload_disabled::<md5>`` placeholder.
        - Otherwise upload the CSV and return the datatalk-dev viewer URL.
        """
        file_path = response.get("file_path")
        if disable_upload_to_azure or not file_path:
            base = response.get("preprocessed_sql") or response.get("generated_sql") or query
            return f"upload_disabled::{hashlib.md5(str(base).encode('utf-8')).hexdigest()}"

        try:
            uploaded = upload_to_azure(file_path)
        except Exception as e:
            _append_failed_query_to_repo_root(
                query=query,
                error="DatatalkRM: upload_to_azure failed",
                rm_name="DatatalkRM",
                debug={"file_path": file_path, "exception": _truncate_for_log(repr(e))},
            )
            # The upload only produces a viewer URL for an answer that has
            # already been computed and written to disk. Raising here unwinds
            # past that answer and makes tree_simulator drop the question
            # entirely ("Skipping failed query"), so an expired SAS token
            # silently discards good retrievals. Fall back to the same
            # placeholder used when uploads are disabled; the CSV is still on
            # disk at `file_path`.
            rm_debug_logger.warning(
                "DatatalkRM: upload_to_azure failed for %s; falling back to "
                "upload_disabled:: placeholder. Error: %s",
                file_path,
                _truncate_for_log(repr(e)),
            )
            base = response.get("preprocessed_sql") or response.get("generated_sql") or query
            return f"upload_disabled::{hashlib.md5(str(base).encode('utf-8')).hexdigest()}"
        viewer = os.getenv("DATATALK_VIEWER_URL", "")
        return (viewer + uploaded) if viewer else uploaded

    def _run_query(
        self,
        *,
        query: str,
        conversation_history: Union[str, list, None],
        disable_upload_to_azure: bool,
        enable_python: bool,
        langfuse_readonly: bool,
        include_summary_stats: bool,
        engine: str,
        designation: Optional[str],
    ) -> dict:
        """Run one query through the in-process library and shape the result."""
        from knowledge_storm.datatalk_agent import library

        parser = self._ensure_parser(langfuse_readonly=langfuse_readonly, engine=engine)

        # The library expects a list[dict]; the Retriever interface threads
        # a JSON-serialized history (string), but tree_simulator may also
        # pass an already-parsed list when it loops through a chain.
        if isinstance(conversation_history, str):
            try:
                conv_hist_list = json.loads(conversation_history) if conversation_history else []
            except json.JSONDecodeError:
                conv_hist_list = []
        elif isinstance(conversation_history, list):
            conv_hist_list = conversation_history
        else:
            conv_hist_list = []

        # Library writes the result JSON + CSV here so the legacy Azure
        # upload path can pick the CSV up.
        os.makedirs(self.SQL_RESULTS_DIR, exist_ok=True)

        with self._call_lock:
            response = asyncio.run(
                library.run_single_message(
                    message=query,
                    conversation_history=conv_hist_list,
                    chainlit_msg_object=None,
                    semantic_parser_class=parser,
                    save_to_local=self.SQL_RESULTS_DIR,
                    save_result_to_csv=True,
                    include_summary_stats=include_summary_stats,
                    designation=designation,
                    engine=engine or "gpt-5",
                )
            )

        url = self._build_url(response, query, disable_upload_to_azure=disable_upload_to_azure)
        return {
            "title": f"SQL Database response for {query}",
            "url": url,
            "snippets": [response.get("summary") or response.get("agent_response", "")],
            "description": response.get("summary") or response.get("agent_response", ""),
            "meta": {
                "SQL": response.get("generated_sql"),
                "preprocessed_sql": response.get("preprocessed_sql"),
                "csv_path": response.get("csv_path"),
                "designation": "SQL",
                "sql_result": response.get("sql_result"),
            },
            "conversation_history": response.get("conversation_history"),
            "result_count": response.get("result_count", 0),
        }

    def forward(
        self,
        query_or_queries: Union[str, List[str]],
        exclude_urls: List[str] = [],
        conversation_history: Union[str, list, None] = None,
        disable_upload_to_azure: bool = None,
        enable_python: bool = None,
        langfuse_readonly: bool = None,
        include_summary_stats: bool = True,
        engine: str = None,
        **kwargs,
    ):
        """
        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        # Resolve per-call overrides against instance defaults.
        effective_disable_upload = self.disable_upload_to_azure if disable_upload_to_azure is None else disable_upload_to_azure
        effective_enable_python = self.enable_python if enable_python is None else enable_python
        effective_langfuse_readonly = self.langfuse_readonly if langfuse_readonly is None else langfuse_readonly
        effective_include_summary_stats = self.include_summary_stats if include_summary_stats is None else include_summary_stats
        effective_engine = self.engine if engine is None else engine
        designation = kwargs.get("designation")

        collected_results = []
        for query in queries:
            try:
                result = self._run_query(
                    query=query,
                    conversation_history=conversation_history,
                    disable_upload_to_azure=effective_disable_upload,
                    enable_python=effective_enable_python,
                    langfuse_readonly=effective_langfuse_readonly,
                    include_summary_stats=effective_include_summary_stats,
                    engine=effective_engine,
                    designation=designation,
                )
            except Exception:
                rm_debug_logger.exception(
                    f"DatatalkRM in-process call failed | domain={self.domain} | query={_truncate_for_log(query)}"
                )
                _append_failed_query_to_repo_root(
                    query=query,
                    error="DatatalkRM: in-process call failed",
                    conversation_history=conversation_history,
                    rm_name="DatatalkRM",
                )
                raise
            collected_results.append(result)

        return collected_results


class DecompositionAgentRM(dspy.Retrieve):
    def __init__(self, k=3, is_valid_source: Callable = None, domain="N/A", wait_time=1200, disable_upload_to_azure: bool = False, enable_python: bool = False, langfuse_readonly: bool = False, include_summary_stats: bool = True, engine: str = None):
        self.usage = 0
        self.domain = domain
        self.rm_type = "datatalk"
        self.wait_time = wait_time
        self.disable_upload_to_azure = disable_upload_to_azure
        self.enable_python = enable_python
        self.langfuse_readonly = langfuse_readonly
        self.include_summary_stats = include_summary_stats
        self.engine = engine
        
        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"YouRM": usage}

    @staticmethod
    def _is_sql_request(kwargs: dict) -> bool:
        return kwargs.get("designation") == "SQL"

    @staticmethod
    def _build_api_url(is_sql: bool) -> str:
        if is_sql:
            return "http://localhost:9527/api/sql/execute"
        return "http://localhost:9527/api/query"

    @staticmethod
    def _build_params(
        query: str,
        is_sql: bool,
        conversation_history: List[dict],
        kwargs: dict,
    ) -> dict:
        if is_sql:
            return {"sql_query": query, **kwargs}
        return {"query": query, "conversation_history": conversation_history, **kwargs}

    @staticmethod
    def _normalize_response_defaults(response: dict) -> None:
        # Defensive defaults: some API error/edge responses may omit these keys.
        response.setdefault("sql_result", None)
        response.setdefault("generated_sql", None)
        response.setdefault("result_count", 0)

    @staticmethod
    def _handle_sql_mode_response(query: str, response: dict) -> None:
        """
        Mutates response in-place to ensure downstream fields exist and `result` is a string.
        """
        result_table = response.get("result_table")

        result_count = response.get("result_count")
        if result_count is None and isinstance(result_table, list):
            result_count = len(result_table)
        if result_count is None:
            result_count = 0

        # Backfill common fields expected downstream
        if isinstance(result_table, list):
            response.setdefault("sql_result", result_table)
        else:
            response.setdefault("sql_result", None)
        response.setdefault("result_count", result_count)

        if result_count > 10:
            summary_stats = _get_table_summary_statistics_from_results_table(result_table)
            if summary_stats:
                response["summary_stats"] = summary_stats
                md = _format_summary_stats_markdown(summary_stats)
                if md:
                    base_text = response.get("result")
                    if base_text is None:
                        base_text = ""
                    if not isinstance(base_text, str):
                        base_text = str(base_text)
                    response["result"] = (base_text.rstrip() + "\n\n" + md).strip() + "\n"
        else:
            if result_table:
                response["result"] = "\n".join([str(row) for row in result_table])
            else:
                response["result"] = "No results found"

        response["generated_sql"] = query
        response["result_count"] = result_count

    @staticmethod
    def _save_result_table_to_csv_and_backfill_response(
        *,
        query: str,
        response: dict,
        conversation_history: List[dict],
    ) -> None:
        """
        Non-SQL mode: if API returns a list of dict rows in `result_table`, persist to CSV and
        populate fields expected by downstream logic.
        """
        result_table = response.get("result_table")
        if not (isinstance(result_table, list) and len(result_table) > 0):
            return

        base_dir = os.getenv(
            "DATASTORM_SQL_RESULTS_DIR",
            str(pathlib.Path(__file__).resolve().parent.parent / "sql_results"),
        )
        os.makedirs(base_dir, exist_ok=True)
        # Build a deterministic, informative filename (let failures raise for debugging)
        base = response.get("generated_sql") or query
        import hashlib

        h = hashlib.md5(str(base).encode("utf-8")).hexdigest()[:10]
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"result_{ts}_{h}.csv"
        saved_path = os.path.join(base_dir, filename)

        # Determine header order by first-seen union of keys
        header_fields = []
        seen = set()
        for row in result_table:
            if isinstance(row, dict):
                for k in row.keys():
                    if k not in seen:
                        seen.add(k)
                        header_fields.append(k)
        with open(saved_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header_fields, extrasaction="ignore")
            writer.writeheader()
            for row in result_table:
                if isinstance(row, dict):
                    writer.writerow({k: row.get(k, "") for k in header_fields})

        # Backfill response fields so downstream logic records/uploads it
        response["csv_path"] = saved_path
        response["file_path"] = saved_path
        response["result_count"] = len(result_table)
        response["sql_result"] = result_table

        if not (response.get("result") is not None and type(response.get("result")) == str):
            _append_failed_query_to_repo_root(
                query=query,
                error="In none SQL mode, result is not a string while success is True, and result table is a list",
                conversation_history=conversation_history,
                rm_name="DecompositionAgentRM",
                debug=response
            )
            response["result"] = "No results found"

    @staticmethod
    def _handle_non_sql_failure(query: str, response: dict, conversation_history: List[dict]) -> None:
        err = response.get("error")
        _append_failed_query_to_repo_root(
            query=query,
            error=str(err),
            conversation_history=conversation_history,
            rm_name="DecompositionAgentRM",
        )
        response["result"] = response.get("error")
        response["generated_sql"] = None
        response["preprocessed_sql"] = None
        response["csv_path"] = None
        response["sql_result"] = None
        response["result_count"] = 0
        assert response["result"] is not None and type(response["result"]) == str

    @staticmethod
    def _handle_non_sql_unexpected_success_shape(
        query: str, response: dict, conversation_history: List[dict]
    ) -> None:
        _append_failed_query_to_repo_root(
            query=query,
            error=f"In non-SQL mode, result table is not a list while success is True.",
            conversation_history=conversation_history,
            rm_name="DecompositionAgentRM",
            debug=response
        )
        response["result"] = str(response)
        response["generated_sql"] = None
        response["preprocessed_sql"] = None
        response["csv_path"] = None
        response["sql_result"] = None
        response["result_count"] = 0

    def _compute_result_url(
        self,
        *,
        query: str,
        response: dict,
        api_url: str,
        http_response: "requests.Response",
        disable_upload_to_azure: bool,
        conversation_history: List[dict],
    ) -> str:
        file_path = (
            response["file_path"] if "file_path" in response else "error_while_saving_datatalk_results"
        )
        if disable_upload_to_azure or file_path == "error_while_saving_datatalk_results":
            base = response.get("preprocessed_sql") or response.get("generated_sql") or query
            import hashlib

            h = hashlib.md5(str(base).encode("utf-8")).hexdigest()
            return f"upload_disabled::{h}"

        try:
            return upload_to_azure(file_path)
        except Exception as e:
            _append_failed_query_to_repo_root(
                query=query,
                error="DecompositionAgentRM: upload_to_azure failed",
                conversation_history=conversation_history,
                rm_name="DecompositionAgentRM",
                debug={
                    "file_path": file_path,
                    "api_url": api_url,
                    "status_code": getattr(http_response, "status_code", None),
                    "exception": _truncate_for_log(repr(e)),
                },
            )
            raise

    @staticmethod
    def _append_conversation_history(conversation_history: List[dict], query: str, response: dict) -> None:
        conversation_history.append(
            {
                "user": query,
                "system": {
                    "textual_result": response["result"],
                    "table": response.get("sql_result"),
                    "sql": response.get("generated_sql"),
                },
            }
        )

    @staticmethod
    def _build_collected_result(
        *,
        query: str,
        url: str,
        response: dict,
        conversation_history: List[dict],
    ) -> dict:
        return {
            "title": f"SQL Database response for {query}",
            "url": url,
            "snippets": [response["result"]],
            "description": response["result"],
            "meta": {
                "SQL": response["generated_sql"],
                "preprocessed_sql": response["generated_sql"],
                "csv_path": response["csv_path"] if "csv_path" in response else None,
                "designation": "SQL",
                "sql_result": response["sql_result"] if "sql_result" in response else None,
            },
            "conversation_history": conversation_history,
            "result_count": response["result_count"],
        }

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = [], conversation_history: List[dict] = [], disable_upload_to_azure: bool = None, enable_python: bool = None, langfuse_readonly: bool = None, **kwargs
    ):
        """
        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        for query in queries:
            try:
                is_sql = self._is_sql_request(kwargs)
                api_url = self._build_api_url(is_sql)
                params = self._build_params(query, is_sql, conversation_history, kwargs)
                headers = {"Content-Type": "application/json"}

                rm_debug_logger.debug(
                    f"POST {api_url} | headers={headers} | json={params} | timeout={self.wait_time}"
                )

                _resp = requests.post(
                    api_url,
                    headers=headers,
                    json=params,
                    timeout=self.wait_time,
                )
                rm_debug_logger.debug(f"Response status={_resp.status_code}")

                response = _resp.json()
                rm_debug_logger.debug(f"Response JSON={response}")

                if isinstance(response, dict):
                    self._normalize_response_defaults(response)

                # Maintain existing behavior: downstream assumes dict-like response.
                result_table = response.get("result_table")

                if is_sql:
                    self._handle_sql_mode_response(query, response)
                elif isinstance(result_table, list):
                    self._save_result_table_to_csv_and_backfill_response(
                        query=query,
                        response=response,
                        conversation_history=conversation_history,
                    )
                elif response.get("success") == False:
                    self._handle_non_sql_failure(query, response, conversation_history)
                else:
                    self._handle_non_sql_unexpected_success_shape(query, response, conversation_history)

                effective_disable_upload = (
                    self.disable_upload_to_azure
                    if disable_upload_to_azure is None
                    else disable_upload_to_azure
                )
                url = self._compute_result_url(
                    query=query,
                    response=response,
                    api_url=api_url,
                    http_response=_resp,
                    disable_upload_to_azure=effective_disable_upload,
                    conversation_history=conversation_history,
                )

                self._append_conversation_history(conversation_history, query, response)
                collected_results.append(
                    self._build_collected_result(
                        query=query,
                        url=url,
                        response=response,
                        conversation_history=conversation_history,
                    )
                )
            except requests.exceptions.Timeout:
                print(
                    f"Timeout for POST {api_url} | json={{'question': query, 'domain': self.domain, 'conversation_history': '...'}} | params={{'api_key': '***'}} | timeout={self.wait_time}"
                )
                raise
            except Exception:
                # Don't swallow exceptions; preserve full stack trace.
                print(
                    f"Exception during POST {api_url} | json={{'question': query, 'domain': self.domain, 'conversation_history': '...'}} | params={{'api_key': '***'}}"
                )
                raise

        return collected_results