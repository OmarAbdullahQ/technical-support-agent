"""Optional Langfuse callback, plus request-correlated local latency/error logs."""
import logging
from functools import lru_cache

from src.config import get_settings

logger = logging.getLogger("support_agent")
logger.setLevel(logging.INFO)


@lru_cache(maxsize=1)
def langfuse_client():
    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(public_key=settings.langfuse_public_key,
                        secret_key=settings.langfuse_secret_key, base_url=settings.langfuse_host)
    except Exception as exc:
        logger.warning("Langfuse initialization disabled: %s", type(exc).__name__)
        return None


def trace_config(request_id: str, trace_id: str) -> dict:
    callbacks = []

    if langfuse_client() is not None:
        try:
            from langfuse.langchain import CallbackHandler

            callbacks.append(
                CallbackHandler(
                    trace_context={"trace_id": trace_id}
                )
            )

        except Exception as exc:
            logger.warning(
                "Langfuse callback disabled: %s",
                type(exc).__name__,
            )

    return {
        "callbacks": callbacks,
        "run_name": "technical-support-request",
        "metadata": {
            "request_id": request_id,
            "trace_id": trace_id,
            "project": "tuwaiq-weekend-support-agent",
        },
    }

def flush():
    client = langfuse_client()
    if client:
        try:
            client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", type(exc).__name__)
