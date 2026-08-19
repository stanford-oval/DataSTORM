import ast
import json
import os
from pydantic import BaseModel
from typing import Union, List, Optional, Literal
from datetime import date

from knowledge_storm.log_utils import logger
from langgraph.graph import END, StateGraph

from knowledge_storm.datatalk_agent.state import Action, DatatalkParserState, SqlQuery, compute_domain_specific_instructions, parse_execute_python_from_sql_output
from knowledge_storm.datatalk_agent.utils import (
    BaseParser,
    format_table_schema,
    get_tables,
    location_linking,
    retrieve_tables_details,
    postprocess_entities,
    RedisLLMCacheEntityLinking
)
from knowledge_storm.datatalk_agent.python_modules.sandbox_runner import execute_python_from_sql_results
from knowledge_storm.langfuse_llm import get_llm, call_llm_with_structured_output

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

REPEATED_ACTION_OBSERVATION = "I have already taken this action. I should not repeat the same action twice. In the next turn I will try something different. If it is impossible to make progress, I should call stop()."

async def json_to_string(j: dict) -> str:
    return json.dumps(j, indent=2, ensure_ascii=False)


async def json_to_action(action_dict: dict) -> Action:
    thought = action_dict["thought"]
    action_name = action_dict["action_name"]
    action_argument = action_dict["action_argument"]

    if action_name == "execute_sql":
        assert action_argument, action_dict

    return Action(
        thought=thought,
        action_name=action_name,
        action_argument=action_argument,
    )

class LLMThoughtAction(BaseModel):
    thought: str
    
    action_name: str # one of 'get_tables', 'retrieve_tables_details', 'execute_sql', 'stop'
    
    action_argument: Union[str, List[str]] # `retrieve_tables_details` argument is a list, others is a str
    
class VerifyDomainSpecificInstructionThoughtActionAction(BaseModel):
    thought: str
    
    action_name: str # one of 'reexecute_sql', 'stop'
    
    action_argument: Optional[str]

async def llmThoughtAction_to_Action(thought_action: LLMThoughtAction) -> Action:
    thought = thought_action.thought
    action_name = thought_action.action_name
    action_argument = thought_action.action_argument

    if action_name in ["execute_sql", "retrieve_tables_details"]:
        assert action_argument

    return Action(
        thought=thought,
        action_name=action_name,
        action_argument=action_argument,
    )

def retrieve_relevant_domain_specific_instructions(
    sql: str,
    instructions: dict,
    controller_reporter = "reporter"
):
    res = []
    for table in instructions.keys():
        if table == "*" or table in sql:
            if controller_reporter == "reporter":
                res.extend(instructions[table]["reporter"])
            elif controller_reporter == "controller":
                res.extend(instructions[table]["controller"])
            else:
                raise ValueError()
    return res

cache_entity_linking = RedisLLMCacheEntityLinking(redis_url="redis://localhost:6379", ttl=3600)

class DatatalkParser(BaseParser):
    def __init__(
        self,
        engine: str,
        table_w_ids: dict,
        database_name: str,
        suql_model_name: str,
        embedding_server_address: str = "http://127.0.0.1:8505",
        free_text_server_address: str = "http://127.0.0.1:8510",
        source_file_mapping: dict = {},
        db_type: Literal["postgres", "sqlite"] = "postgres",
        db_secrets_file = None,
        examples: list | None = None,
        table_schema: list | None = None,
        domain_specific_instructions: str | None = None,
        available_actions: list | None = [
            "get_tables", # retrieve relevant tables based on a query
        ],
        suql_enabled = False,
        enable_python = False,
        langfuse_readonly: bool = False,
        num_init_steps_cached = 2,
        db_path = None
    ):
        async def initialize_state(input):
            question = input["question"]
            (f"Started processing {question}")
            
            return DatatalkParserState(
                question=question,
                conversation_history=input["conversation_history"],
                engine=engine,
                action_counter=0,
                total_action_counter=0,
                actions=[],
                examples=examples or [],
                table_schemas=table_schema,
                dlg_history=[],
                domain_specific_instructions=compute_domain_specific_instructions(domain_specific_instructions),
                verify_domain_specific_instructions_counter=0,
                db_type=db_type,
                suql_enabled=suql_enabled,
                enable_python=enable_python,
                langfuse_readonly=langfuse_readonly,
                db_secrets_file=db_secrets_file,
                database_name=database_name,
                num_init_steps_cached=num_init_steps_cached,
                available_actions=available_actions,
                table_w_ids=table_w_ids,
                embedding_server_address=embedding_server_address,
                free_text_server_address=free_text_server_address,
                source_file_mapping=source_file_mapping,
                suql_model_name=suql_model_name,
                entity_linking_results=input["entity_linking_results"] if "entity_linking_results" in input else {},
                db_path=db_path,
            )
        
        self.runnable = initialize_state | DatatalkParser.compiled_graph
        self.database_name = database_name
        self.langfuse_readonly = langfuse_readonly
        
    
    @classmethod
    def initialize(
        cls,
        engine: str
    ):
        # build the graph
        graph = StateGraph(DatatalkParserState)
        graph.add_node("controller", DatatalkParser.controller)
        graph.add_node("execute_sql", DatatalkParser.execute_sql)
        graph.add_node("get_tables", DatatalkParser.get_tables)
        graph.add_node("retrieve_tables_details", DatatalkParser.retrieve_tables_details)
        graph.add_node("entity_linking", DatatalkParser.entity_linking)
        graph.add_node("location_linking", DatatalkParser.location_linking)
        graph.add_node("stop", DatatalkParser.stop)
        graph.add_node("verify_domain_specific_instructions", DatatalkParser.verify_domain_specific_instructions)
        graph.add_node("execute_python_from_sql", DatatalkParser.execute_python_from_sql)

        graph.set_entry_point("controller")

        graph.add_conditional_edges(
            "controller",
            DatatalkParser.router,  # the function that will determine which node is called next.
        )
        for n in [
            "execute_sql",
            "get_tables",
            "retrieve_tables_details",
            "entity_linking",
            "location_linking",
            "execute_python_from_sql"
        ]:
            graph.add_edge(n, "controller")
        graph.add_edge("stop", "verify_domain_specific_instructions")
        
        # router decides whether to end
        graph.add_conditional_edges(
            "verify_domain_specific_instructions",
            DatatalkParser.router,
        )
        
        cls.compiled_graph = graph.compile()

    @staticmethod
    async def sql_chain(
        sql: str,
        state,
        limit_query = " LIMIT 1000"
    ) -> SqlQuery:
        sql_query_object = SqlQuery(
            sql=sql,
            table_w_ids=state["table_w_ids"] if "table_w_ids" in state else {},
            database_name=state["database_name"],
            embedding_server_address=state["embedding_server_address"],
            free_text_server_address=state.get("free_text_server_address", "http://127.0.0.1:8510"),
            source_file_mapping=state["source_file_mapping"],
            suql_model_name=state["suql_model_name"],
            db_type=state["db_type"],
            db_secrets_file=state["db_secrets_file"],
            suql_enabled=state["suql_enabled"],
            db_path=state["db_path"]
        )
        await sql_query_object.execute(limit_query=limit_query)
        return sql_query_object
        
    @staticmethod
    def get_current_action(state):
        return state["actions"][-1]

    @staticmethod
    async def router(state):
        if state["verify_domain_specific_instructions_counter"] > 2:
            # logger.info("Run out of verification %s", state["question"])
            return END
        # chitchatting
        elif len(state["actions"]) == 1 and state["actions"][-1].action_name == "stop":
            return END
        elif len(state["actions"]) > 2 and state["actions"][-2].action_name == "stop" and state["actions"][-1].action_name == "stop":
            return END

        elif state["action_counter"] >= 15 or state["total_action_counter"] >= 21:
            return END
        
        elif state["actions"][-1].action_name == "error":
            return END

        actions = state["actions"]
        # check if REPEATED_ACTION_OBSERVATION appears twice in a row in the last two actions
        if len(actions) >= 3:
            if state["actions"][-1].action_name == "execute_sql" and state["actions"][-2].action_name == "execute_sql":
                if state["actions"][-2].observation == REPEATED_ACTION_OBSERVATION and state["actions"][-1].action_argument == state["actions"][-2].action_argument:
                    return END
        
        return state["actions"][-1].action_name

    @staticmethod
    async def controller(state):
        async def get_cached_initial_actions(cached_step_num):
            if cached_step_num == 1:
                first_action_thought = "To answer the question, I need to identify the tables containing the relevant information. I'll start by retrieving the available tables to understand the database structure."
                first_action_name = "get_tables"
                action = Action(
                    thought=first_action_thought, 
                    action_name=first_action_name, 
                    action_argument=None,
                )

            elif cached_step_num == 2:
                second_action_thought = "I need to retrieve the details of the each table in the database to understand their structure and identify relevant tables and columns to answer the question."
                second_action_name = "retrieve_tables_details"
                second_action_arg = [table_dict['table_name'] for table_dict in get_tables(state['table_schemas'])]
                action = Action(
                    thought=second_action_thought, 
                    action_name=second_action_name, 
                    action_argument=second_action_arg,
                )
            return action
        
        # make the history shorter
        actions = state["actions"][-10:]
        action_history = []
        image_dicts = []
        
        get_tables_schema_results = []
        for i, a in enumerate(actions):
            include_observation = True
            
            if i < len(actions) - 4 and a.action_name in [
                "get_tables",
            ]:
                include_observation = False
            elif i < len(actions) - 3 and a.action_name in [
                "execute_sql",
            ]:
                include_observation = False
                
            # exclude get_tables_schema results that are the same
            if a.action_name == "get_tables_schema":
                if a.observation in get_tables_schema_results:
                    include_observation = False
                else:
                    get_tables_schema_results.append(a.observation)
            
            action_history.append(a.to_jinja_string(include_observation))
            if a.images_b64s:
                image_dicts=[{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                } for image_data in a.images_b64s]
        
        # if a previous turn is to execute a SQL that involved one of the
        # domain specific instructions, update it here
        domain_specific_instructions = []
        if len(actions) >= 1 and actions[-1].action_name in ["execute_sql"]:
            domain_specific_instructions = retrieve_relevant_domain_specific_instructions(
                actions[-1].action_argument,
                state["domain_specific_instructions"],
                controller_reporter="controller"
            )
        else:
            domain_specific_instructions = retrieve_relevant_domain_specific_instructions(
                "",
                state["domain_specific_instructions"],
                controller_reporter="controller"
            )
    
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if state["conversation_history"] == [] and not state["actions"] and state["num_init_steps_cached"] > 0:
                    action = await get_cached_initial_actions(1)
                elif state["conversation_history"] == [] and len(state["actions"]) == 1 and state["num_init_steps_cached"] > 1:
                    action = await get_cached_initial_actions(2)

                else:
                    action = await call_llm_with_structured_output(
                        "datatalk_controller",
                        {
                            "question": state["question"],
                            "action_history": action_history,
                            "conversation_history": state["conversation_history"],
                            "domain_specific_instructions": domain_specific_instructions,
                            "database_type": state["db_type"],
                            "suql_enabled": state["suql_enabled"],
                            "enable_python": state["enable_python"],
                            "curr_date": date.today(),
                            "available_actions": state["available_actions"],
                        },
                        LLMThoughtAction,
                        get_llm(),
                        image_dicts=image_dicts,
                        langfuse_readonly=state["langfuse_readonly"],
                    )
                    action = await llmThoughtAction_to_Action(action)
                break
            except Exception as e:
                # Log the error if needed
                print(f"controller chain encountered the following unexpected error: {e}")
                if attempt == max_retries - 1:
                    action = Action(
                        "I encountered an error producing a thought",
                        "error",
                        ""
                    )
            
        res_actions = state["actions"]
        res_actions.append(action)
        return {"actions": res_actions, "action_counter": 1, "total_action_counter": 1}

    @staticmethod
    async def execute_sql(state):
        current_action = DatatalkParser.get_current_action(state)
        assert current_action.action_name == "execute_sql"

        # populate dictionary with substrings to substitute exactly with value in post-processing
        # for now, this is just the results of entity_linking
        substitution_dict = state["entity_linking_results"]
        current_action.postprocessed_action_argument = await postprocess_entities(current_action.action_argument, substitution_dict)
        sql = await DatatalkParser.sql_chain(current_action.postprocessed_action_argument, state)
        current_action.postprocessed_action_argument = (
            sql.sql
        ) # update it with the cleaned and optimized SQL


        if sql.has_results():
            current_action.observation = sql.execution_result_sample
        else:
            current_action.observation = sql.execution_status
        current_action.result_count = sql.result_count

        # move_back_on_duplicate_action = 1
        def check_sql_in_last_two_turns(current_action, state):
            for i in [2,3]:
                if len(state["actions"]) >= i and state["actions"][-i].action_name == "execute_sql" and current_action.action_argument == state["actions"][-i].action_argument:
                    state["actions"][-1].observation = REPEATED_ACTION_OBSERVATION
                    return state
            return state
        
        current_action = DatatalkParser.get_current_action(state)
        if current_action.action_name == "execute_sql" :
            state = check_sql_in_last_two_turns(current_action, state)

        return {"generated_sqls": sql, "actions": state["actions"]}

    @staticmethod
    async def get_tables(state):
        current_action = DatatalkParser.get_current_action(state)
        assert current_action.action_name == "get_tables"
        action_results = get_tables(
            state["table_schemas"]
        )
        current_action.observation = format_table_schema(action_results)

    # INCOMPLETE
    @staticmethod
    async def entity_linking(state):
        current_action = DatatalkParser.get_current_action(state)
        assert current_action.action_name == "entity_linking"
        raw_arg = current_action.action_argument

        def parse_entity_linking_args(raw_arg):
            try:
                if isinstance(raw_arg, str):
                    # Convert string input to tuple
                    parsed_tuple = eval(raw_arg)
                else:
                    parsed_tuple = raw_arg

                # Check other formatting errors
                if not isinstance(parsed_tuple, tuple) and not isinstance(parsed_tuple, list):
                    return "Invalid input format - the input to entity_linking must be formatted as a Python tuple (enclosed in () parenthesis)", 1
                if len(parsed_tuple) != 2:
                    return "Invalid input format - the tuple input to entity_linking must contain two arguments - a search string and a list of columns.", 1
                
                search_str, entity_table_nm = parsed_tuple
                
                # Remove quotes from search_str if they are present at both ends
                if (search_str.startswith("'") and search_str.endswith("'")) or \
                   (search_str.startswith('"') and search_str.endswith('"')):
                    search_str = search_str[1:-1]
                
            except Exception:
                return "Invalid input format - the input to entity_linking must be formatted as a Python tuple, but some other format was provided.", 1
            
            ### Now we have a Tuple input ###
            # Validate search string
            if not isinstance(search_str, str):
                return "Invalid input format - the first argument to entity_linking must be a single Python string.", 1
                
            # Validate col_name_lst
            if not isinstance(entity_table_nm, str):
                return "Invalid input format - the second argument to entity_linking must be a single Python string.", 1
         
            return (search_str, entity_table_nm), 0

        def entity_list_to_filtering_set(entity_list, entity_table_nm, search_str):
            var_name = "".join(search_str.split()) + entity_table_nm  # construct deterministic variable name
            var_name = var_name.replace(",", "")
            var_name = var_name.replace("'", "")
            var_name = var_name.replace('"', "")
            var_name = var_name.replace('-', "")
            entity_set_str = ["'" + ent.replace("'", "''") + "'" for ent in entity_list]  # Escape apotrophe/quotes and encase in single-quote
            if not entity_set_str:
                return var_name, None
            entity_set_str = f"({', '.join(entity_set_str)})"
            return var_name, entity_set_str


        # For simplicity + compatibility with pervious code, LLM is supplying 1 string argument that is post-processed here
        parse_res, parse_error = parse_entity_linking_args(raw_arg)

        if parse_error:
            # Send parsing error back to controller
            result = parse_res
        else:
            search_str, entity_table_nm = parse_res
            ent_lst = await cache_entity_linking.entity_linking_caching(
                search_str,
                entity_table_nm,
                state["db_secrets_file"],
                langfuse_readonly=state["langfuse_readonly"]
            )
            if type(ent_lst) is str:
                current_action.observation = ent_lst
                return
            
            result = ""
            var_name, var_entities = entity_list_to_filtering_set(ent_lst, entity_table_nm, search_str)
            if var_entities is None:
                result += f"No matching entities from {entity_table_nm} on the string '{search_str}' found.\n"
            else:
                state["entity_linking_results"][var_name] = var_entities  # store variable in state
                result += f"{var_name} now stores the list of resolved entities from {entity_table_nm} on the string '{search_str}'.\n"
                result += "I will use these variables in future filtering clauses."

        current_action.observation = result

    
    @staticmethod
    async def location_linking(state):
        current_action = DatatalkParser.get_current_action(state)
        assert current_action.action_name == "location_linking"

        raw_arg = current_action.action_argument        
        try:
        # Check if raw_arg is already a list
            if isinstance(raw_arg, list):
                loc_lst = raw_arg
            else:
                # Try parsing as a Python literal
                loc_lst = ast.literal_eval(raw_arg)
        except (ValueError, SyntaxError):
            try:
                loc_lst = json.loads(raw_arg)
            except (json.JSONDecodeError, TypeError):
                raise ValueError(f"Invalid input format for location linking: {raw_arg}")

        assert type(loc_lst) is list

        action_results = await location_linking(
            loc_lst,
            langfuse_readonly=state["langfuse_readonly"]
        )

        current_action.observation = str(action_results)

        # For front-end to clearly indicate the action
        current_action.action_argument = f"location_linking({str(loc_lst)})"

    
    @staticmethod
    async def retrieve_tables_details(state):
        current_action = DatatalkParser.get_current_action(state)
        assert current_action.action_name == "retrieve_tables_details"
        action_results = retrieve_tables_details(
            current_action.action_argument,
            state["table_schemas"],
        )
        current_action.observation = format_table_schema(action_results)
        

    @staticmethod
    async def verify_domain_specific_instructions(state):
        """
        Verify that all domain-specific instructions have been adhered to.
        """
        include_observation_actions = ['entity_linking', 'location_linking']
        
        if not state["generated_sqls"]:
            # no generated sql exists. no need to verify anything
            return {"verify_domain_specific_instructions_counter": 3}
        
        # Get the last non-postprocessed generated SQL to pass into verifier
        sql = next(
            (act.action_argument for act in reversed(state["actions"]) if act.action_name == 'execute_sql'),
            None
        )

        domain_specific_instructions = retrieve_relevant_domain_specific_instructions(
            sql,
            state["domain_specific_instructions"],
            controller_reporter="controller"
        )

        actions = state["actions"][-10:]
        action_history = []
        for i, a in enumerate(actions):
            include_observation_bool = True if a.action_name in include_observation_actions else False
            action_history.append(a.to_jinja_string(include_observation_bool))

        action = await call_llm_with_structured_output(
            "datatalk_verify_domain_specific_instructions",
            {
                "question": state["question"],
                "conversation_history": state["conversation_history"],
                "sql": sql,
                "domain_specific_instructions": domain_specific_instructions,
                "database_type": state["db_type"],
                "suql_enabled": state["suql_enabled"],
                "action_history": action_history
            },
            VerifyDomainSpecificInstructionThoughtActionAction,
            get_llm(),
            langfuse_readonly=state["langfuse_readonly"],
        )
        
        if action.action_name == "reexecute_sql":
            thought = action.thought
            action_name = "execute_sql"
            action_argument = action.action_argument

            action = Action(
                thought=thought,
                action_name=action_name,
                action_argument=action_argument,
            )
            # do not count this as an extra action yet
            res_actions = state["actions"][:-2]
            res_actions.append(action)
            return {"actions": res_actions, "verify_domain_specific_instructions_counter": 1}
        
        else:
            # logger.info("verify_domain_specific_instructions: Finished run for question %s", state["question"])
            action = Action(
                thought=action.thought,
                action_name=action.action_name,
                action_argument=action.action_argument,
            )
            res_actions = state["actions"]
            res_actions.append(action)
            return {"actions": res_actions, "final_sql": sql}

    @staticmethod
    async def execute_python_from_sql(state):
        current_action = DatatalkParser.get_current_action(state)
        assert current_action.action_name == "execute_python_from_sql"
        
        def parse_execute_python_from_sql_args(raw_arg):
            try:
                if isinstance(raw_arg, str):
                    # Convert string input to tuple
                    parsed_tuple = eval(raw_arg)
                else:
                    parsed_tuple = raw_arg

                # Check other formatting errors
                if not isinstance(parsed_tuple, tuple) and not isinstance(parsed_tuple, list):
                    return "Invalid input format - the input to execute_python_from_sql must be formatted as a Python tuple (enclosed in () parenthesis)", 1
                if len(parsed_tuple) != 2:
                    return "Invalid input format - the tuple input to execute_python_from_sql must contain two arguments - a SQL query and a Python code.", 1
                
                sql, python_code = parsed_tuple
                
            except Exception:
                return "Invalid input format - the input to entity_linking must be formatted as a Python tuple, but some other format was provided.", 1
            
            if not isinstance(sql, str):
                return "Invalid input format - the first argument to execute_python_from_sql must be a single Python string.", 1
                
            if not isinstance(python_code, str):
                return "Invalid input format - the second argument to execute_python_from_sql must be a single Python string.", 1
         
            return (sql, python_code), 0
        
        parse_res, parse_error = parse_execute_python_from_sql_args(current_action.action_argument)

        if parse_error:
            # Send parsing error back to controller
            current_action.observation = parse_res
            current_action.images_b64s = []
            current_action.postprocessed_action_argument = parse_res
        else:
            sql, python_code = parse_res
            substitution_dict = state["entity_linking_results"]
            current_action.postprocessed_action_argument = await postprocess_entities(current_action.action_argument, substitution_dict)
            sql = await DatatalkParser.sql_chain(sql, state)
            result = execute_python_from_sql_results(sql.get_results_dataframe(), python_code)
            cleaned, images_b64s = parse_execute_python_from_sql_output(result)
        
            current_action.observation = cleaned
            current_action.images_b64s = images_b64s
            current_action.postprocessed_action_argument = f"""sql_results = execute_sql("{sql.sql}")\n\n{python_code}"""

    @staticmethod
    async def stop(state):
        current_action = DatatalkParser.get_current_action(state)
        assert current_action.action_name == "stop"
        for s in state["generated_sqls"]:
            assert isinstance(s, SqlQuery)
