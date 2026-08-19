"""LLM interaction utilities.

Helper functions for working with language models.
"""

import asyncio
import contextlib
import hashlib
import json
import re
from typing import Any, Callable, Optional, TypeVar

import redis.asyncio as redis
from knowledge_storm.log_utils import logger
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langfuse import observe
from pydantic import BaseModel

from knowledge_storm.langfuse_prompts import compile_prompt, compile_prompt_jinja


"""LLM model instances and factory functions.

Provides access to configured language model instances.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_openai import ChatOpenAI
import os

from knowledge_storm.utils import load_api_key
current_dir = os.path.dirname(os.path.abspath(__file__))
secrets_path = os.path.join(current_dir, '..', 'secrets.toml')
load_api_key(toml_file_path=os.path.abspath(secrets_path))

def get_llm(
    model_name: str = "gpt-5",
    rate_limiter: BaseRateLimiter = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Get LLM with appropriate temperature based on completions.

    Args:
        completions: How many completions we need

    Returns:
        Configured LLM instance
    """
    # Ensure trailing slash: LangChain's ChatOpenAI instance-client normalizes,
    # but match the convention used elsewhere so the env var is uniformly handled.
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is not set. Copy secrets_example.toml to "
            "secrets.toml and fill in your model endpoint and API key."
        )
    base = endpoint.rstrip("/") + "/"
    return ChatOpenAI(
        openai_api_base=base,
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        rate_limiter=rate_limiter,
    )

T = TypeVar("T")
R = TypeVar("R")
M = TypeVar("M", bound=BaseModel)


# Redis client for caching
_redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> Optional[redis.Redis]:
    """Get or create Redis client for caching."""
    global _redis_client
    if _redis_client is not None:
        # Verify the client is still usable (asyncio.run() closes the event loop each call,
        # which invalidates clients created in a previous loop).
        try:
            await _redis_client.ping()
        except Exception:
            _redis_client = None
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host="localhost",
                port=6379,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            # Test connection
            await _redis_client.ping()
            logger.info("Connected to Redis for LLM caching")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Caching disabled.")
            _redis_client = None
    return _redis_client


def _get_llm_identifier(llm: BaseChatModel) -> str:
    """
    Best-effort identifier for the LLM to use in cache keys.

    Heuristics:
    - If instance of AzureChatOpenAI, prefer its deployment_name.
    - Else try common attributes in order: model, model_name, name, id.
    - Fallback to the fully qualified class name.
    """
    # AzureChatOpenAI special-case without a hard dependency
    try:
        from langchain_openai import AzureChatOpenAI  # type: ignore

        if isinstance(llm, AzureChatOpenAI):  # pragma: no cover - optional dependency
            deployment = getattr(llm, "deployment_name", None)
            if deployment:
                return f"azure:{deployment}"
            # fallback to model attribute if present
            model_attr = getattr(llm, "model", None)
            if model_attr:
                return f"azure:{model_attr}"
    except Exception:
        pass

    # Generic fallbacks
    for attr in ("model", "model_name", "name", "id"):
        value = getattr(llm, attr, None)
        if value:
            return str(value)

    # Final fallback: fully qualified class name
    cls = llm.__class__
    return f"{cls.__module__}.{cls.__name__}"


def _generate_cache_key(
    prompt_content: str, output_class_name: str, llm: BaseChatModel, tools_fingerprint: Optional[str] = None, image_dicts: Optional[list[dict[str, Any]]] = None
) -> str:
    """Generate a cache key including prompt, model identity, output class, and tools fingerprint."""
    model_identifier = _get_llm_identifier(llm)
    fingerprint = tools_fingerprint or ""
    content = f"{prompt_content}:{model_identifier}:{output_class_name}:{fingerprint}:{image_dicts}"
    return f"llm_cache:{hashlib.md5(content.encode()).hexdigest()}"


def _fingerprint_tools(tools: Optional[list[Any]]) -> str:
    """
    Create a deterministic, order-independent fingerprint of the provided tools for caching.

    Why
    - When tools influence model behavior, their identity must be part of the cache key to avoid
      collisions between otherwise-identical prompts that differ only by available tools.

    Input
    - tools: Optional list of callables or LangChain tools. If None or empty, returns an empty string.

    Method
    - For each tool, collect stable metadata:
      - name/__name__
      - module and qualname
      - call signature (via inspect.signature)
      - args_schema JSON if present (supports both Pydantic v1 .schema() and v2 .model_json_schema())
      - MD5 of the tool's source code when available (best-effort)
    - Build per-tool descriptors, sort them by name for order-independence, JSON-serialize with
      sorted keys, then return MD5 of that serialized payload.

    Properties
    - Deterministic across runs given the same tool definitions.
    - Order-independent with respect to the input list.
    - Best-effort: if inspection fails, falls back to hashing repr(tool).
    - Does not execute tools and is safe for side-effectful implementations.
    - If source code is unavailable (e.g., C extensions, dynamic callables), fingerprint remains
      stable based on available metadata only.

    Example:
    For the `web_search` tool, a single-tool descriptor may look like:
        [
            {
                "module": "langchain_core.tools.structured",
                "name": "web_search",
                "schema": {"description": "Search the web for the given queries...", "...": "..."},
                "signature": "(tool_input: 'str', callbacks: 'Callbacks' = None) -> 'str'",
                "source_hash": null
            }
        ]

    Returns
    - Hex MD5 string representing the tool set fingerprint, or an empty string when no tools are
      provided or fingerprinting ultimately fails.
    """
    if not tools:
        return ""
    try:
        import inspect

        tool_descriptors: list[dict[str, Any]] = []
        for tool in tools:
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None) or repr(tool)
            module = getattr(tool, "__module__", None)
            qualname = getattr(tool, "__qualname__", None)
            try:
                signature = str(inspect.signature(tool))
            except Exception:
                signature = ""

            schema_json = None
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None:
                try:
                    if hasattr(args_schema, "model_json_schema"):
                        schema_json = args_schema.model_json_schema()
                    elif hasattr(args_schema, "schema"):
                        schema_json = args_schema.schema()
                except Exception:
                    schema_json = None

            try:
                source = inspect.getsource(tool)
                source_hash = hashlib.md5(source.encode()).hexdigest()
            except Exception:
                source_hash = None

            tool_descriptors.append(
                {
                    "name": str(name),
                    "module": str(module),
                    "qualname": str(qualname),
                    "signature": signature,
                    "schema": schema_json,
                    "source_hash": source_hash,
                }
            )

        # Sort by name for determinism
        tool_descriptors.sort(key=lambda d: d.get("name", ""))
        serialised = json.dumps(tool_descriptors, sort_keys=True, default=str)
        return hashlib.md5(serialised.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Tool fingerprinting failed: {e}")
        # Fallback to hash of repr of tools
        try:
            serialised = json.dumps([repr(t) for t in tools], sort_keys=True, default=str)
            return hashlib.md5(serialised.encode()).hexdigest()
        except Exception:
            return ""


def _serialise_message_for_model(message: Any) -> dict[str, Any]:
    """Best-effort conversion of LangChain message objects to dicts for model input and cache keys."""
    try:
        if isinstance(message, AIMessage):
            # Extract tool_calls if present
            tool_calls = getattr(message, "tool_calls", None)
            return {
                "role": "assistant",
                "content": getattr(message, "content", ""),
                "tool_calls": tool_calls,
            }
        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "content": getattr(message, "content", ""),
                "tool_call_id": getattr(message, "tool_call_id", None),
            }
        if isinstance(message, dict):
            return message
        # Unknown type fallback
        return {"role": getattr(message, "role", "assistant"), "content": str(message)}
    except Exception:
        return {"role": "assistant", "content": str(message)}


def _normalise_for_cache_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise messages for cache key computation by removing volatile fields.

    - For assistant tool_calls, keep only name and args (drop generated ids or other metadata)
    - For tool messages, drop tool_call_id
    """
    normalised: list[dict[str, Any]] = []
    for m in messages:
        try:
            role = m.get("role")
            content = m.get("content")
            nm: dict[str, Any] = {"role": role, "content": content}
            if role == "assistant" and "tool_calls" in m:
                raw_calls = m.get("tool_calls") or []
                simplified_calls = []
                for c in raw_calls:
                    try:
                        # Support both dict and attr styles
                        name = getattr(c, "name", None) if not isinstance(c, dict) else c.get("name")
                        args = getattr(c, "args", None) if not isinstance(c, dict) else c.get("args")
                        simplified_calls.append({"name": name, "args": args})
                    except Exception:
                        simplified_calls.append({})
                nm["tool_calls"] = simplified_calls
            elif role == "tool":
                # Keep content only for determinism
                pass
            else:
                # Keep other fields if present but ensure determinism by sorting keys via json later
                for k, v in m.items():
                    if k not in ("role", "content"):
                        nm[k] = v
            normalised.append(nm)
        except Exception:
            normalised.append({"role": m.get("role", "assistant"), "content": m.get("content")})
    return normalised


async def compile_prompt_from_template(*_args, **_kwargs) -> list[dict[str, str]]:  # pragma: no cover
    """Deprecated: local prompt templates are no longer supported."""
    raise RuntimeError("Local prompt templates are no longer supported. All prompts must be fetched from Langfuse.")


async def call_llm_with_structured_output(
    prompt_id: str,
    variables: dict[str, any],
    output_class: type[M],
    llm: BaseChatModel,
    tools: Optional[list[Any]] = None,
    context_desc: str = "",
    force_tool_first_turn: str = "auto",
    use_cache: bool = True,
    langfuse_readonly: bool = False,
    extra_messages: Optional[list[Any]] = None,
    image_dicts: Optional[list[dict[str, Any]]] = None,
) -> Optional[M]:
    """Call LLM with structured output, trace prompt & model via Langfuse.

    Uses Jinja2 compilation by default with fallback to original compilation method.
    This supports full Jinja2 templating including for loops, conditionals, filters, etc.

    Args:
        prompt_id: prompt group name
        variables: variables for compiling prompt (supports any type, not just strings)
        output_class: pydantic model
        llm: chat model instance
        tools: optional list of tools for a first-turn tool call
        context_desc: log string
        force_tool_first_turn: tool choice for the first turn when tools are provided
        return_first_ai_message_on_tools: when tools are provided, the raw first AIMessage (tool call turn) is returned
        use_cache: whether to use Redis caching
        langfuse_readonly: when True, do not upload anything to Langfuse; only read prompts

    Returns:
        - With tools: the raw AIMessage containing tool calls
        - Without tools: the structured output (pydantic model) or None if error
    """

    # If read-only mode is requested, use the non-upload implementation and return early
    if langfuse_readonly:
        return await call_llm_with_structured_output_readonly(
            prompt_id=prompt_id,
            variables=variables,
            output_class=output_class,
            llm=llm,
            tools=tools,
            context_desc=context_desc,
            force_tool_first_turn=force_tool_first_turn,
            use_cache=use_cache,
            extra_messages=extra_messages,
            image_dicts=image_dicts,
        )

    @observe(name=prompt_id, as_type="generation")
    async def _call_llm_with_structured_output(
        prompt_id: str,
        variables: dict[str, any],
        output_class: type[M],
        llm: BaseChatModel,
        tools: Optional[list[Any]] = None,
        context_desc: str = "",
        force_tool_first_turn: str = "auto",
        use_cache: bool = True,
        extra_messages: Optional[list[Any]] = None,
        image_dicts: Optional[list[dict[str, Any]]] = None, # callers are responsible for providing the images in the correct format, see https://python.langchain.com/docs/how_to/multimodal_inputs/
    ) -> Optional[M]:
        # Try Jinja2 compilation first, fallback to original method if it fails
        prompt_obj = None
        message_tuples = None
        # compilation_method = "jinja2"

        try:
            prompt_obj, message_tuples = compile_prompt_jinja(prompt_id, variables)
        except Exception as jinja_error:
            logger.warning(
                f"Jinja2 compilation failed for {context_desc}: {jinja_error}. Falling back to original method."
            )
            # compilation_method = "original"

            # Fallback to original compilation method
            try:
                # Convert variables to strings for original method compatibility
                str_variables = {}
                for key, value in variables.items():
                    if isinstance(value, str):
                        str_variables[key] = value
                    else:
                        str_variables[key] = str(value)

                prompt_obj, message_tuples = compile_prompt(prompt_id, str_variables)
            except Exception as original_error:
                logger.error(
                    f"Both Jinja2 and original prompt compilation failed for {context_desc}. "
                    f"Jinja2 error: {jinja_error}. Original error: {original_error}"
                )
                return None

        if not message_tuples:
            logger.error(f"No message tuples generated for {context_desc}")
            return None

        try:
            messages = [{"role": role, "content": content} for role, content in message_tuples]
            if image_dicts:
                messages.append({"role": "user", "content": [image_dict for image_dict in image_dicts]})

            # Append any extra messages (e.g., AI tool-call and ToolMessage results)
            if extra_messages:
                try:
                    serialised_extras = [_serialise_message_for_model(m) for m in extra_messages]
                    messages.extend(serialised_extras)
                except Exception as e:
                    logger.warning(f"Failed to serialise extra_messages for {context_desc}: {e}")

            # Generate cache key based on the compiled prompt content
            messages_for_cache = _normalise_for_cache_messages(messages)
            prompt_content_serialised = json.dumps(messages_for_cache, sort_keys=True)
            tools_fp = _fingerprint_tools(tools) if tools else None
            cache_key = _generate_cache_key(prompt_content_serialised, output_class.__name__, llm, tools_fp, image_dicts)
        except Exception as e:
            logger.error(f"Message processing failed for {context_desc}: {e}")
            return None

        # gpt-5 disables temperature; point releases (gpt-5.x) use a different effort scale
        if (llm.model_name and "gpt-5" in llm.model_name) or (getattr(llm, "deployment_name", None) and "gpt-5" in getattr(llm, "deployment_name", "")):
            llm.temperature = 1.0
            _model_id = llm.model_name or getattr(llm, "deployment_name", "") or ""
            _m = re.search(r"gpt-5\.(\d+)", _model_id)
            llm.reasoning_effort = "low" if (_m and int(_m.group(1)) >= 4) else "minimal"

        # Check cache first if enabled and we have a redis connection
        # Caching is disabled for non-zero temperature models because their outputs are non-deterministic
        # due to the randomness introduced by the temperature setting. Caching such outputs could lead
        # to inconsistent or misleading results.
        if llm.temperature == 0.0 or (llm.model_name and "gpt-5" in llm.model_name) or (getattr(llm, "deployment_name", None) and "gpt-5" in getattr(llm, "deployment_name", "")):  # pragma: no cover
            redis_client = await get_redis_client()
        else:
            redis_client = None
            logger.debug(f"Cache disabled for {context_desc} because temperature is {llm.temperature}")

        if redis_client and use_cache:
            try:
                cached_result = await redis_client.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for {context_desc}")
                    cached_data = json.loads(cached_result)
                    # Support both structured outputs and tool-call AI messages
                    if isinstance(cached_data, dict) and "kind" in cached_data:
                        kind = cached_data.get("kind")
                        payload = cached_data.get("data")
                        if kind == "structured":
                            return output_class.model_validate(payload)
                        if kind == "ai_message":
                            try:
                                from langchain_core.messages import AIMessage as _AIMessage

                                content = (payload or {}).get("content", "")
                                tool_calls = (payload or {}).get("tool_calls")
                                ai = _AIMessage(content=content, tool_calls=tool_calls)
                                # best-effort restore for any additional kwargs
                                additional_kwargs = (payload or {}).get("additional_kwargs")
                                if additional_kwargs is not None:
                                    with contextlib.suppress(Exception):
                                        ai.additional_kwargs = additional_kwargs
                                return ai  # type: ignore[return-value]
                            except Exception as e:
                                logger.warning(f"Failed to reconstruct AIMessage from cache: {e}")
                    # Backwards compatibility: assume structured payload
                    return output_class.model_validate(cached_data)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        try:
            # link prompt & model to current generation
            try:
                from langfuse import get_client  # local import to avoid mandatory dep

                client = get_client()
                # Prompts loaded from the local snapshot have no server-side ID,
                # so they cannot be linked to a generation.
                if prompt_obj is not None and not getattr(prompt_obj, "is_local", False):
                    client.update_current_generation(prompt=prompt_obj, model=MODEL_NAME)  # type: ignore[attr-defined]
                else:
                    client.update_current_generation(model=MODEL_NAME)  # type: ignore[attr-defined]
            except Exception:
                pass

            final_result = None
            result_kind = None

            # If tools are provided, perform a single-call tool turn and try to parse structured output from it
            if tools:
                # First turn: bind tools and request tool calls
                try:
                    # Use only the base messages for the initial tool call; extra_messages apply to follow-up calls
                    base_messages = []
                    try:
                        # Rebuild base messages from tuples to avoid including extra messages in the first turn
                        base_messages = [{"role": role, "content": content} for role, content in message_tuples]
                    except Exception:
                        base_messages = list(messages)

                    llm_with_tools = llm.bind_tools(tools, tool_choice=force_tool_first_turn)
                    # Disable LangChain's internal cache and callbacks for this call; we manage caching ourselves and avoid tracer serialization issues.
                    first_ai = await llm_with_tools.ainvoke(base_messages, config={"cache": False, "callbacks": []})
                    if not isinstance(first_ai, AIMessage):
                        logger.error(f"Expected AIMessage on first turn for {context_desc}")
                        return None
                    final_result = first_ai
                    result_kind = "ai_message"
                except Exception as e:
                    logger.error(f"Initial LLM (tool-call) invoke failed for {context_desc}: {e}")
                    return None
            else:
                # No tools path: single structured output call with fallback to JSON mode
                try:
                    runner = llm.with_structured_output(output_class)
                except Exception as e_primary:
                    try:
                        runner = llm.with_structured_output(output_class, method="json_mode")
                    except Exception as e_json:
                        logger.error(
                            f"with_structured_output failed for {context_desc}: {e_primary}; "
                            f"json_mode fallback also failed: {e_json}"
                        )
                        return None

                # Disable LangChain's internal cache and callbacks for this call; we manage caching ourselves and avoid tracer serialization issues.
                llm_result = await runner.ainvoke(messages, config={"cache": False, "callbacks": []})

                # Convert dict result to Pydantic model if needed
                result = output_class.model_validate(llm_result) if isinstance(llm_result, dict) else llm_result
                final_result = result
                result_kind = "structured"

            # Cache the result if successful (single exit point)
            if final_result and use_cache and cache_key:
                redis_client = await get_redis_client()
                if redis_client:
                    try:
                        payload = None
                        if result_kind == "structured":
                            try:
                                payload = {"kind": "structured", "data": final_result.model_dump()}
                            except Exception:
                                # fallback if model_dump is unavailable
                                payload = {
                                    "kind": "structured",
                                    "data": json.loads(json.dumps(final_result, default=str)),
                                }
                        elif result_kind == "ai_message":
                            try:
                                payload = {
                                    "kind": "ai_message",
                                    "data": {
                                        "content": getattr(final_result, "content", ""),
                                        "tool_calls": getattr(final_result, "tool_calls", None),
                                        "additional_kwargs": getattr(final_result, "additional_kwargs", None),
                                    },
                                }
                            except Exception:
                                payload = None

                        if payload is not None:
                            # Cache for 1 hour
                            await redis_client.setex(cache_key, 3600, json.dumps(payload, default=str))
                            logger.debug(f"Cached result for {context_desc}")
                    except Exception as e:
                        logger.warning(f"Cache write error: {e}")

            return final_result
        except Exception as e:
            logger.error(f"LLM call failed for {context_desc}: {e}")
            return None

    return await _call_llm_with_structured_output(
        prompt_id=prompt_id,
        variables=variables,
        output_class=output_class,
        llm=llm,
        tools=tools,
        context_desc=context_desc,
        force_tool_first_turn=force_tool_first_turn,
        use_cache=use_cache,
        extra_messages=extra_messages,
        image_dicts=image_dicts,
    )


async def call_llm_with_structured_output_readonly(
    prompt_id: str,
    variables: dict[str, any],
    output_class: type[M],
    llm: BaseChatModel,
    tools: Optional[list[Any]] = None,
    context_desc: str = "",
    force_tool_first_turn: str = "auto",
    use_cache: bool = True,
    extra_messages: Optional[list[Any]] = None,
    image_dicts: Optional[list[dict[str, Any]]] = None,
) -> Optional[M]:
    """Call LLM with structured output WITHOUT uploading anything to Langfuse.
    
    - Still compiles the prompt from Langfuse (read-only).
    - No @observe decorator and no client.update_current_generation calls.
    - Preserves caching and tool-call behavior.
    """
    # Compile prompt (read-only from Langfuse)
    prompt_obj = None
    message_tuples = None
    try:
        prompt_obj, message_tuples = compile_prompt_jinja(prompt_id, variables)
    except Exception as jinja_error:
        logger.warning(
            f"Jinja2 compilation failed for {context_desc}: {jinja_error}. Falling back to original method."
        )
        try:
            # Convert variables to strings for original method compatibility
            str_variables = {}
            for key, value in variables.items():
                if isinstance(value, str):
                    str_variables[key] = value
                else:
                    str_variables[key] = str(value)
            prompt_obj, message_tuples = compile_prompt(prompt_id, str_variables)
        except Exception as original_error:
            logger.error(
                f"Both Jinja2 and original prompt compilation failed for {context_desc}. "
                f"Jinja2 error: {jinja_error}. Original error: {original_error}"
            )
            return None

    if not message_tuples:
        logger.error(f"No message tuples generated for {context_desc}")
        return None

    # Build messages and cache key
    try:
        messages = [{"role": role, "content": content} for role, content in message_tuples]
        if image_dicts:
            messages.append({"role": "user", "content": [image_dict for image_dict in image_dicts]})

        if extra_messages:
            try:
                serialised_extras = [_serialise_message_for_model(m) for m in extra_messages]
                messages.extend(serialised_extras)
            except Exception as e:
                logger.warning(f"Failed to serialise extra_messages for {context_desc}: {e}")

        messages_for_cache = _normalise_for_cache_messages(messages)
        prompt_content_serialised = json.dumps(messages_for_cache, sort_keys=True)
        tools_fp = _fingerprint_tools(tools) if tools else None
        cache_key = _generate_cache_key(
            prompt_content_serialised, output_class.__name__, llm, tools_fp, image_dicts
        )
    except Exception as e:
        logger.error(f"Message processing failed for {context_desc}: {e}")
        return None

    # Model-specific adjustments (keep parity with original)
    # Point releases (gpt-5.x) use a different reasoning_effort scale from base gpt-5.
    if (llm.model_name and "gpt-5" in llm.model_name) or (getattr(llm, "deployment_name", None) and "gpt-5" in getattr(llm, "deployment_name", "")):
        llm.temperature = 1.0
        _model_id = llm.model_name or getattr(llm, "deployment_name", "") or ""
        _m = re.search(r"gpt-5\.(\d+)", _model_id)
        llm.reasoning_effort = "low" if (_m and int(_m.group(1)) >= 4) else "minimal"

    # Cache read (deterministic models only)
    if llm.temperature == 0.0 or (llm.model_name and "gpt-5" in llm.model_name) or (getattr(llm, "deployment_name", None) and "gpt-5" in getattr(llm, "deployment_name", "")):  # pragma: no cover
        redis_client = await get_redis_client()
    else:
        redis_client = None
        logger.debug(f"Cache disabled for {context_desc} because temperature is {llm.temperature}")

    if redis_client and use_cache:
        try:
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for {context_desc}")
                cached_data = json.loads(cached_result)
                if isinstance(cached_data, dict) and "kind" in cached_data:
                    kind = cached_data.get("kind")
                    payload = cached_data.get("data")
                    if kind == "structured":
                        return output_class.model_validate(payload)
                    if kind == "ai_message":
                        try:
                            from langchain_core.messages import AIMessage as _AIMessage
                            content = (payload or {}).get("content", "")
                            tool_calls = (payload or {}).get("tool_calls")
                            ai = _AIMessage(content=content, tool_calls=tool_calls)
                            additional_kwargs = (payload or {}).get("additional_kwargs")
                            if additional_kwargs is not None:
                                with contextlib.suppress(Exception):
                                    ai.additional_kwargs = additional_kwargs
                            return ai  # type: ignore[return-value]
                        except Exception as e:
                            logger.warning(f"Failed to reconstruct AIMessage from cache: {e}")
                # Backwards compatibility: assume structured payload
                return output_class.model_validate(cached_data)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

    # Perform the call without any Langfuse uploads or tracing
    try:
        final_result = None
        result_kind = None

        if tools:
            # First turn tool-call only
            try:
                base_messages = []
                try:
                    base_messages = [{"role": role, "content": content} for role, content in message_tuples]
                except Exception:
                    base_messages = list(messages)

                llm_with_tools = llm.bind_tools(tools, tool_choice=force_tool_first_turn)
                first_ai = await llm_with_tools.ainvoke(base_messages, config={"cache": False, "callbacks": []})
                if not isinstance(first_ai, AIMessage):
                    logger.error(f"Expected AIMessage on first turn for {context_desc}")
                    return None
                final_result = first_ai
                result_kind = "ai_message"
            except Exception as e:
                logger.error(f"Initial LLM (tool-call) invoke failed for {context_desc}: {e}")
                return None
        else:
            # Structured output single call
            try:
                runner = llm.with_structured_output(output_class)
            except Exception as e_primary:
                try:
                    runner = llm.with_structured_output(output_class, method="json_mode")
                except Exception as e_json:
                    logger.error(
                        f"with_structured_output failed for {context_desc}: {e_primary}; "
                        f"json_mode fallback also failed: {e_json}"
                    )
                    return None
            llm_result = await runner.ainvoke(messages, config={"cache": False, "callbacks": []})
            result = output_class.model_validate(llm_result) if isinstance(llm_result, dict) else llm_result
            final_result = result
            result_kind = "structured"

        # Cache write
        if final_result and use_cache and cache_key:
            redis_client = await get_redis_client()
            if redis_client:
                try:
                    payload = None
                    if result_kind == "structured":
                        try:
                            payload = {"kind": "structured", "data": final_result.model_dump()}
                        except Exception:
                            payload = {"kind": "structured", "data": json.loads(json.dumps(final_result, default=str))}
                    elif result_kind == "ai_message":
                        try:
                            payload = {
                                "kind": "ai_message",
                                "data": {
                                    "content": getattr(final_result, "content", ""),
                                    "tool_calls": getattr(final_result, "tool_calls", None),
                                    "additional_kwargs": getattr(final_result, "additional_kwargs", None),
                                },
                            }
                        except Exception:
                            payload = None
                    if payload is not None:
                        await redis_client.setex(cache_key, 3600, json.dumps(payload, default=str))
                        logger.debug(f"Cached result for {context_desc}")
                except Exception as e:
                    logger.warning(f"Cache write error: {e}")

        return final_result
    except Exception as e:
        logger.error(f"LLM call failed for {context_desc}: {e}")
        return None


async def call_llm_with_structured_output_cached(
    prompt_id: str,
    variables: dict[str, any],
    output_class: type[M],
    llm: BaseChatModel,
    context_desc: str = "",
    use_cache: bool = True,
    langfuse_readonly: bool = False,
) -> Optional[M]:
    """Cached version of call_llm_with_structured_output.

    This is an alias for backwards compatibility.
    """
    return await call_llm_with_structured_output(
        prompt_id=prompt_id,
        variables=variables,
        output_class=output_class,
        llm=llm,
        context_desc=context_desc,
        use_cache=True,
        langfuse_readonly=langfuse_readonly,
    )


async def call_llm_with_tools(
    prompt_id: str,
    variables: dict[str, Any],
    output_class: type[M],
    llm: BaseChatModel,
    tools: list[Any],
    context_desc: str = "",
    force_tool_first_turn: str = "auto",
) -> Optional[M]:
    """Call LLM with tools, forcing a first-turn tool call then returning final structured output.

    Flow:
    1) First turn (tool call): delegate to call_llm_with_structured_output with tools provided
    2) Execute tools and append ToolMessage results
    3) Second turn (structured): delegate to call_llm_with_structured_output with extra messages and no tools
    """

    # First turn: expect a tool call via the unified helper
    try:
        first_ai = await call_llm_with_structured_output(
            prompt_id=prompt_id,
            variables=variables,
            output_class=output_class,
            llm=llm,
            tools=tools,
            context_desc=context_desc,
            force_tool_first_turn=force_tool_first_turn,
            use_cache=True,
        )
    except Exception as e:
        logger.error(f"Initial tool-call turn failed for {context_desc}: {e}")
        return None

    if not isinstance(first_ai, AIMessage):
        logger.error(f"Expected AIMessage on first turn for {context_desc}")
        return None

    tool_calls = getattr(first_ai, "tool_calls", None)
    if not tool_calls:
        logger.error(f"Model did not produce tool calls for {context_desc}")
        return None

    # Build a name->tool mapping for execution
    name_to_tool: dict[str, Any] = {}
    try:
        for t in tools:
            tool_name = getattr(t, "name", None) or getattr(t, "__name__", None)
            if tool_name:
                name_to_tool[tool_name] = t
    except Exception:
        pass

    # Execute tool calls sequentially and build extra messages to append
    extra_messages: list[Any] = [first_ai]
    for tc in tool_calls:
        try:
            call_name = getattr(tc, "name", None) or tc.get("name")
            call_args = getattr(tc, "args", None) or tc.get("args", {})
            call_id = getattr(tc, "id", None) or tc.get("id", "tool_call")

            tool = name_to_tool.get(str(call_name))
            if tool is None:
                logger.error(f"Requested tool '{call_name}' not provided for {context_desc}")
                return None

            tool_output = await tool.ainvoke(call_args)
            extra_messages.append(ToolMessage(tool_call_id=call_id, content=str(tool_output)))
        except Exception as e:
            logger.error(f"Tool execution error for {context_desc}: {e}")
            return None

    # Second turn: ask for structured output using the same helper, passing extra_messages
    try:
        final_result = await call_llm_with_structured_output(
            prompt_id=prompt_id,
            variables=variables,
            output_class=output_class,
            llm=llm,
            tools=None,
            context_desc=context_desc,
            force_tool_first_turn="auto",
            use_cache=True,
            extra_messages=extra_messages,
        )
        return final_result
    except Exception as e:
        logger.error(f"Final LLM structured invoke failed for {context_desc}: {e}")
        return None


async def process_with_voting(
    items: list[T],
    processor: Callable[[T, Any], tuple[bool, Optional[R]]],
    llm: Any,
    completions: int,
    min_successes: int,
    result_factory: Callable[[R, T], Any],
    description: str = "item",
) -> list[Any]:
    """Process items with multiple LLM attempts and consensus voting.

    Args:
        items: Items to process
        processor: Function that processes each item
        llm: LLM instance
        completions: How many attempts per item
        min_successes: How many must succeed
        result_factory: Function to create final result
        description: Item type for logs

    Returns:
        list of successfully processed results
    """
    results = []

    for item in items:
        # Make multiple attempts
        attempts = await asyncio.gather(*[processor(item, llm) for _ in range(completions)])

        # Count successes
        success_count = sum(1 for success, _ in attempts if success)

        # Only proceed if we have enough successes
        if success_count < min_successes:
            logger.info(f"Not enough successes ({success_count}/{min_successes}) for {description}")
            continue

        # Use the first successful result
        for success, result in attempts:
            if success and result is not None:
                processed_result = result_factory(result, item)
                if processed_result:
                    results.append(processed_result)
                    break

    return results

if __name__ == "__main__":
    from datetime import datetime
    from knowledge_storm.utils import load_api_key
    load_api_key(toml_file_path=os.path.join(os.path.dirname(__file__), "../secrets.toml"))

    class SummarizeModel(BaseModel):
        """Summary of results"""
        summary: str
    import base64
    
    with open(os.getenv("DATASTORM_TEST_IMAGE", "test_plot.png"), "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    result = asyncio.run(call_llm_with_structured_output(
        "plot_interpretation",
        {},
        SummarizeModel,
        get_llm(),
        image_dicts=[{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_data}"
            }
        }]
    ))
    print(result)