import base64
import csv
import json
import operator
import re
from collections import defaultdict
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

import pandas as pd

from knowledge_storm.datatalk_agent.sql_utils import execute_sql

# `chainlit` is imported lazily inside print_chainlit() — importing it at
# module load triggers chainlit's app-instantiation side effects (creating
# .chainlit/ in the cwd, loading translation files, etc.), which is hostile
# to library use. Library callers that don't run under chainlit never call
# print_chainlit, so the lazy import keeps them clean.

pd.set_option('display.float_format', '{:.2f}'.format)  # 2 decimal places


def json_to_panda_markdown(
    data: List[dict],
    head = 10,
    processing_fcn = None
) -> str:
    df = pd.DataFrame(data)
    
    # Determine the number of rows to show at the top and bottom
    num_head = head // 2
    num_tail = head - num_head
    
    # Create the truncated DataFrame
    if len(df) > head and head > 0:
        head_df = df.head(num_head)
        tail_df = df.tail(num_tail)
        
        # Number of omitted rows
        omitted_rows = len(df) - head
        
        # Create a custom row with '... (x omitted) | ... | ...'
        custom_row = pd.DataFrame({col: ['...'] for col in df.columns})
        custom_row[df.columns[0]] = f'... ({omitted_rows} omitted)'
        
        # Concatenate the head, custom row, and tail DataFrames
        truncated_df = pd.concat([head_df, custom_row, tail_df], ignore_index=True)
    else:
        truncated_df = df

    if processing_fcn:
        truncated_df = processing_fcn(truncated_df)

    for col in truncated_df.select_dtypes(include=["int"]).columns:
        truncated_df[col] = truncated_df[col].map(lambda x: f"{x:,.0f}")

    for col in truncated_df.select_dtypes(include=["float"]).columns:
        truncated_df[col] = truncated_df[col].map(lambda x: f"{x:,.2f}")

    return truncated_df.to_markdown(index=False)


MAX_RESULT_TOKENS = 2000

def json_to_panda_markdown_token_limited(
    data: List[dict],
    max_tokens: int = MAX_RESULT_TOKENS,
    processing_fcn=None,
) -> str:
    import tiktoken

    df = pd.DataFrame(data)

    if processing_fcn:
        df = processing_fcn(df)

    for col in df.select_dtypes(include=["int"]).columns:
        df[col] = df[col].map(lambda x: f"{x:,.0f}")
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = df[col].map(lambda x: f"{x:,.2f}")

    enc = tiktoken.get_encoding("o200k_base")

    full_md = df.to_markdown(index=False)
    if len(enc.encode(full_md)) <= max_tokens:
        return full_md

    # Split rendered markdown into header lines and per-row lines
    lines = full_md.split('\n')
    header_lines = lines[:2]   # column names + separator line
    data_lines = lines[2:]     # one line per data row

    header_tokens = len(enc.encode('\n'.join(header_lines)))
    omit_tokens = 20  # rough budget for the "... (X omitted)" row
    row_token_counts = [len(enc.encode(line)) for line in data_lines]

    budget = max_tokens - header_tokens - omit_tokens
    half = budget // 2

    # Greedily fill head rows
    head_count, head_used = 0, 0
    for t in row_token_counts:
        if head_used + t <= half:
            head_used += t
            head_count += 1
        else:
            break

    # Greedily fill tail rows
    tail_count, tail_used = 0, 0
    for t in reversed(row_token_counts[head_count:]):
        if tail_used + t <= half:
            tail_used += t
            tail_count += 1
        else:
            break

    omitted = len(df) - head_count - tail_count
    if omitted <= 0:
        return full_md

    head_df = df.head(head_count)
    tail_df = df.tail(tail_count)
    omit_row = pd.DataFrame({col: ['...'] for col in df.columns})
    omit_row[df.columns[0]] = f'... ({omitted} omitted)'
    truncated_df = pd.concat([head_df, omit_row, tail_df], ignore_index=True)
    return truncated_df.to_markdown(index=False)


def json_to_panda_csv(
    data: List[dict],
    processing_fcn = None
) -> str:
    df = pd.DataFrame(data)
    if processing_fcn:
        df = processing_fcn(df)
    return df.to_csv(index=False)


def json_to_markdown_table(data):
    # Assuming the JSON data is a list of dictionaries
    # where each dictionary represents a row in the table
    if not data or not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("JSON data is not in the expected format for a table")

    def rearrange_list(lst):
        if "_id" in lst:
            lst.remove("_id")
            lst.insert(0, "_id")
        return lst

    # Extract headers, and put _id up front
    headers = rearrange_list(list(data[0].keys()))
    
    # Start building the Markdown table
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["-" * len(header) for header in headers]) + " |\n"

    # Add rows
    for row in data:
        markdown_table += "| " + " | ".join(str(row[header]) for header in headers) + " |\n"

    return markdown_table


def convert_json_to_table_format(data):
    # Load the JSON data
    if isinstance(data, str):
        data = json.loads(data)

    # Convert the JSON data to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Convert the DataFrame to a table format
    table = df.to_markdown(index=False)

    return table


def parse_execute_python_from_sql_output(output: str) -> tuple[str, list[str]]:
    """
    Parse the stdout/stderr produced by execute_python_from_sql_results.

    Extracts inline base64-encoded PNG images emitted between
    <<KRAKEN_PLOT:PNG>>...<<END>> markers and removes those segments from text.

    Also normalizes any plot error sentinels <<KRAKEN_PLOT_ERROR>>...<<END>>
    into a readable bracketed note in the cleaned text.

    Returns:
        cleaned_text: str - the original text with plot markers removed
        images_b64: list[str] - list of base64 strings for each extracted image
    """
    if not output:
        return "", []

    # Extract images and remove their markers from text
    plot_pattern = re.compile(r"<<KRAKEN_PLOT:PNG>>(.*?)<<END>>", re.DOTALL)
    images_b64 = [m.group(1).strip() for m in plot_pattern.finditer(output)]
    cleaned_text = plot_pattern.sub("", output)

    # Replace plot error markers with a readable note
    err_pattern = re.compile(r"<<KRAKEN_PLOT_ERROR>>(.*?)<<END>>", re.DOTALL)
    cleaned_text = err_pattern.sub(lambda m: f"[Plot error: {m.group(1)}]", cleaned_text)

    cleaned_text = cleaned_text.strip()
    if not cleaned_text and images_b64:
        cleaned_text = f"{len(images_b64)} plot is generated"

    return cleaned_text, images_b64


class SqlQuery:
    def __init__(
        self,
        sql: Optional[str],
        table_w_ids: dict,
        database_name: str,
        embedding_server_address: str,
        free_text_server_address: str,
        source_file_mapping: dict,
        suql_model_name: str,
        db_type: str,
        db_secrets_file: str,
        suql_enabled: bool,
        db_path: Optional[str] = None
    ):
        if "SELECT" not in sql:
            self.is_valid = False
        self.sql = SqlQuery.clean_sql(sql)
        self.table_w_ids = table_w_ids
        self.database_name = database_name
        self.embedding_server_address = embedding_server_address
        self.free_text_server_address = free_text_server_address
        self.source_file_mapping = source_file_mapping
        self.is_valid = True
        self.result_count = -1
        self.db_path = db_path

        self.execution_result_sample = None
        self.execution_result_full_dict = None
        self.execution_status = None

        self.suql_model_name = suql_model_name

        self.db_type = db_type
        self.db_secrets_file = db_secrets_file
        self.suql_enabled = suql_enabled
        # self.full_result_enums = full_result_enums

    @staticmethod
    def clean_sql(sql: str):
        if sql is None:
            return sql
        cleaned_sql = sql.strip()
        return cleaned_sql


    async def execute(
        self,
        processing_fcn = None,
        limit_query = " LIMIT 1000"
    ):
        if self.sql == "SELECT * FROM information_schema.tables WHERE table_schema = 'public';":
            self.execution_result_sample = "Prohibited. Use `get_tables_schema`"
            self.execution_result_full_dict = "Prohibited. Use `get_tables_schema`"
            return
        
        # TODO: probably need to perform some post processing by using column_names with self.execution_result
        execution_result, self.execution_status = await execute_sql(
            self.sql,
            self.table_w_ids,
            self.database_name,
            self.suql_model_name,
            self.embedding_server_address,
            self.free_text_server_address,
            self.source_file_mapping,
            self.db_type,
            self.db_secrets_file,
            self.suql_enabled,
            limit_query = limit_query,
            db_path = self.db_path
        )


        if execution_result is not None:
            # if self.sql in self.full_result_enums:
            #     self.execution_result_sample = json_to_panda_markdown(dict_result, head=-1)
            # else:
            
            head = 10
            # TODO: hardcoded
            if "SELECT * FROM transaction_type_codes" in self.sql:
                head = -1
            
            self.execution_result_sample = json_to_panda_markdown(
                execution_result,
                head = head,
                processing_fcn = processing_fcn
            )
            self.execution_result_full_dict = execution_result
            self.result_count = len(execution_result)

    def get_table_summary_statistics(self):
        if self.execution_result_full_dict is None:
            return None
        
        if not self.execution_result_full_dict:  # Empty result
            return {}
        
        df = pd.DataFrame(self.execution_result_full_dict)
        stats = {}

        def _stringify_unhashable(v):
            """
            pandas' nunique()/value_counts() require hashable values. SQL results can include
            nested JSON-like objects (dict/list) which are unhashable in Python.
            """
            if isinstance(v, (dict, list, set, tuple)):
                try:
                    return json.dumps(v, sort_keys=True, default=str)
                except Exception:
                    return str(v)
            return v
        
        for column in df.columns:
            col_stats = {}
            
            # Number of distinct values
            series = df[column]
            hashable_series = series.map(_stringify_unhashable)
            distinct_values = hashable_series.nunique()
            # Percentage of distinct values
            total_values = len(df)
            distinct_percentage = (distinct_values / total_values * 100) if total_values > 0 else 0
            col_stats["distinct_percentage"] = f"{round(distinct_percentage, 2)}%"
            
            # Check column type for min/max values
            sample_value = series.iloc[0] if not series.empty else None
            if sample_value is not None:
                # Check if column contains numeric or date/time values
                if isinstance(sample_value, (int, float, pd.Timestamp, pd.Timedelta)) or pd.api.types.is_numeric_dtype(series):
                    try:
                        col_stats["min"] = series.min()
                        col_stats["max"] = series.max()
                        # Add median and mean for numeric columns
                        try:
                            col_stats["median"] = series.median()
                            col_stats["mean"] = series.mean()
                        except:
                            pass  # Skip if median/mean calculation fails
                    except:
                        pass  # Skip if min/max fails
            
            # Top 5 values and their counts
            if distinct_percentage < 100:
                value_counts = hashable_series.value_counts().head(5).to_dict()
                col_stats["top_values"] = [{"value": k, "count": v} for k, v in value_counts.items()]
            
            stats[column] = col_stats
        
        return stats


    def get_results_dataframe(self):
        """
        Convert the full execution result (list of dicts) into a pandas DataFrame.

        Returns:
            pd.DataFrame | None: A DataFrame built from `execution_result_full_dict`,
            or None if no results are available.
        """
        if self.execution_result_full_dict is None:
            return None
        return pd.DataFrame(self.execution_result_full_dict)


    def has_results(self) -> bool:
        return self.execution_result_sample is not None

    def __repr__(self):
        return f"Sql({self.sql})"

    def __hash__(self):
        return hash(self.sql)


def merge_dictionaries(dictionary_1: dict, dictionary_2: dict) -> dict:
    """
    Merges two dictionaries, combining their key-value pairs.
    If a key exists in both dictionaries, the value from dictionary_2 will overwrite the value from dictionary_1.

    Parameters:
        dictionary_1 (dict): The first dictionary.
        dictionary_2 (dict): The second dictionary.

    Returns:
        dict: A new dictionary containing the merged key-value pairs.
    """
    merged_dict = dictionary_1.copy()  # Start with a copy of the first dictionary
    merged_dict.update(
        dictionary_2
    )  # Update with the second dictionary, overwriting any duplicates
    return merged_dict


def merge_sets(set_1: set, set_2: set) -> set:
    return set_1 | set_2


def add_item_to_list(_list: list, item) -> list:
    ret = _list.copy()
    # if item not in ret:
    ret.append(item)
    return ret


class BaseParserState(TypedDict):
    question: str
    engine: str
    generated_sqls: Annotated[list[SqlQuery], add_item_to_list]
    final_sql: SqlQuery
    action_counter: Annotated[int, operator.add]
    total_action_counter: Annotated[int, operator.add]
    examples: list[str]
    table_schemas: list[dict[str, str]]
    conversation_history: list
    domain_specific_instructions: dict
    response: str
    verify_domain_specific_instructions_counter: Annotated[int, operator.add]
    db_type: str
    suql_enabled: bool
    enable_python: bool
    langfuse_readonly: bool
    db_secrets_file: str
    database_name: str
    last_timer: float
    num_init_steps_cached: int
    available_actions: list
    table_w_ids: dict
    embedding_server_address: str
    free_text_server_address: str
    source_file_mapping: str
    suql_model_name: str
    entity_linking_results: dict
    db_path: Optional[str]


class Action:
    possible_actions = [
        "get_tables",
        "retrieve_tables_details",
        "execute_sql",
        "get_examples",
        "entity_linking",
        "location_linking",
        "stop",
        "execute_python_from_sql",

        "error" # a special action to denote that an error occured in one of the action parsing
    ]

    # All actions have a single input parameter for now
    def __init__(self, thought: str, action_name: str, action_argument: str, observation: str = None, images_b64s: List[str] = []):
        if action_name.strip() == "stop()":
            action_name = "stop"
        
        self.thought = thought
        self.action_name = action_name
        self.action_argument = action_argument
        self.observation = observation
        self.postprocessed_action_argument = action_argument  # for now - it may become different from standard action_argument if entity_linking gets called
        self.result_count = None # this only applies to execute SQLs.
        self.images_b64s = images_b64s

        assert self.action_name in Action.possible_actions, f"{self.action_name} not in permitted actions"

    # Necessary for serializing action history for rule proposal generation
    def to_dict(self) -> Dict[str, Any]:
        """Convert Action to dictionary for JSON serialization"""
        return {
            "thought": self.thought,
            "action_name": self.action_name,
            "action_argument": self.action_argument,
            "observation": self.observation,
            "images_b64s": self.images_b64s
        }
        
    @classmethod
    def from_dict(cls, data):
        """Dynamically instantiate an object from a dictionary."""
        return cls(**data)

    def to_jinja_string(self, include_observation: bool) -> str:
        if not self.observation:
            observation = "Did not find any results."
        else:
            observation = self.observation
        ret = f"Thought: {self.thought}\nAction: {self.action_name}({self.action_argument})\n"
        if include_observation:
            ret += f"Observation: {observation}\n"
        else:
            # for SQL queries, we still append a string showing how long the output was
            if self.result_count is not None:
                if self.result_count == -1:
                    ret += f"Did not find any results"
                else:
                    ret += f"Observation omitted due to length. Retrieved {self.result_count} rows\n"
            ret += f"Observation omitted due to length.\n"
            
        return ret
    
    async def print_chainlit(self, step_name):
        import chainlit as cl  # lazy: see module-level note about chainlit side effects
        async with cl.Step(name=step_name, type="tool", show_input=False) as step_thought:
            thought = self.thought
            action = f"{self.postprocessed_action_argument}"
            
            if not self.action_argument:
                 step_thought.output = f"""### Thought
{thought}
"""

            elif not self.observation:
                step_thought.output = f"""### Thought
{thought}

### Action
```sql\n{action}
```
"""
                
            elif self.action_name in ['entity_linking', 'location_linking']:
                observation = self.observation
                observation="```python\n"+str(observation)+"\n```"
                
                step_thought.output = f"""### Thought
{thought}

### Action
```python\n{action}
```

### Observation
{observation}
"""
            elif self.action_name == "execute_python_from_sql":
                cleaned, images_b64 = self.observation, self.images_b64s
                image_elements = []
                for i, b64_data in enumerate(images_b64):
                    try:
                        img_bytes = base64.b64decode(b64_data)
                        image_elements.append(
                            cl.Image(
                                content=img_bytes,
                                mime="image/png",
                                name=f"plot_{i+1}.png",
                                display="inline",
                            )
                        )
                    except Exception:
                        # If decoding fails, skip this image but keep text
                        pass
                if image_elements:
                    step_thought.elements = image_elements

                observation_md = f"```python\n{cleaned}\n```" if cleaned else ""

                step_thought.output = f"""### Thought
{thought}

### Action
```python
{action}
```

### Observation
{observation_md}
            """

            else:
                observation = self.observation
                if self.action_name == "get_tables_schema":
                    observation="```python\n"+str(observation)+"\n```"
            
                step_thought.output = f"""### Thought
{thought}

### Action
```sql\n{action}
```

### Observation
{observation}
"""

    def __repr__(self) -> str:
        if not self.observation:
            observation = "Did not find any results."
        else:
            observation = self.observation
        return f"Thought: {self.thought}\nAction: {self.action_name}({self.action_argument})\nObservation: {observation}"

    def __eq__(self, other):
        if not isinstance(other, Action):
            return NotImplemented
        return (
            self.action_name == other.action_name
            and self.action_argument == other.action_argument
        )

    def __hash__(self):
        return hash((self.action_name, self.action_argument))


class DatatalkParserState(BaseParserState):
    actions: Sequence[Action]

def compute_domain_specific_instructions(csv_file_path):
    if not csv_file_path:
        return dict()
    
    res = defaultdict(lambda: {
        "reporter": [],
        "controller": [],
    })
    with open(csv_file_path, "r") as fd:
        reader = csv.DictReader(fd)
    
        for row in reader:
            # there are 2 formats for this file for now.
            
            # One using simple 3 row headers: table_name, instruction, report_controller_flag
            # Another one from the rule-proposing framework.
            if "trigger_condition" in row.keys():
                table_name_row_name = "trigger_condition"
            else:
                table_name_row_name = "table_name"
            
            if "report_controller_flag" not in row or row["report_controller_flag"] == "0":
                res[row[table_name_row_name]]["reporter"].append(row["instruction"])
                res[row[table_name_row_name]]["controller"].append(row["instruction"])
            elif row["report_controller_flag"] == "1":
                res[row[table_name_row_name]]["reporter"].append(row["instruction"])
            elif row["report_controller_flag"] == "2":
                res[row[table_name_row_name]]["controller"].append(row["instruction"])
            else:
                raise ValueError()
    return res

def locate_last_generated_base64_image(action_history: list[Action], return_llm_compatible_dict: bool = False) -> List[str]:
    for a in reversed(action_history):
        if a.images_b64s:
            if return_llm_compatible_dict:
                return [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                } for image_data in a.images_b64s]
            else:
                return a.images_b64s
    return []
