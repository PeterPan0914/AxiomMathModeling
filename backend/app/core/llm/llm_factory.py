"""LLM factory module, creates LLM instances for each Agent based on config."""

from app.config.setting import settings
from app.core.llm.llm import LLM


def _resolve(per_value, default_value):
    """Fall back to LLM_DEFAULT_* shared default when per-Agent field is empty.

    Args:
        per_value: Per-Agent config value (e.g. COORDINATOR_API_KEY).
        default_value: Shared default (LLM_DEFAULT_*).

    Returns:
        Effective value.
    """
    if per_value not in (None, ""):
        return per_value
    return default_value


class LLMFactory:
    """LLM factory: creates coordinator, modeler, coder, writer LLM instances.

    Priority: per-Agent field > LLM_DEFAULT_* shared default.
    """

    task_id: str

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id

    def get_all_llms(self) -> tuple[LLM, LLM, LLM, LLM]:
        """Create all four agent LLM instances.

        Returns:
            (coordinator_llm, modeler_llm, coder_llm, writer_llm) tuple.
        """
        coordinator_llm = LLM(
            api_type=_resolve(settings.COORDINATOR_API_TYPE, settings.LLM_DEFAULT_API_TYPE),
            api_key=_resolve(settings.COORDINATOR_API_KEY, settings.LLM_DEFAULT_API_KEY),
            model=_resolve(settings.COORDINATOR_MODEL, settings.LLM_DEFAULT_MODEL),
            base_url=_resolve(settings.COORDINATOR_BASE_URL, settings.LLM_DEFAULT_BASE_URL),
            task_id=self.task_id,
            max_tokens=_resolve(settings.COORDINATOR_MAX_TOKENS, settings.LLM_DEFAULT_MAX_TOKENS),
        )

        modeler_llm = LLM(
            api_type=_resolve(settings.MODELER_API_TYPE, settings.LLM_DEFAULT_API_TYPE),
            api_key=_resolve(settings.MODELER_API_KEY, settings.LLM_DEFAULT_API_KEY),
            model=_resolve(settings.MODELER_MODEL, settings.LLM_DEFAULT_MODEL),
            base_url=_resolve(settings.MODELER_BASE_URL, settings.LLM_DEFAULT_BASE_URL),
            task_id=self.task_id,
            max_tokens=_resolve(settings.MODELER_MAX_TOKENS, settings.LLM_DEFAULT_MAX_TOKENS),
        )

        coder_llm = LLM(
            api_type=_resolve(settings.CODER_API_TYPE, settings.LLM_DEFAULT_API_TYPE),
            api_key=_resolve(settings.CODER_API_KEY, settings.LLM_DEFAULT_API_KEY),
            model=_resolve(settings.CODER_MODEL, settings.LLM_DEFAULT_MODEL),
            base_url=_resolve(settings.CODER_BASE_URL, settings.LLM_DEFAULT_BASE_URL),
            task_id=self.task_id,
            max_tokens=_resolve(settings.CODER_MAX_TOKENS, settings.LLM_DEFAULT_MAX_TOKENS),
        )

        writer_llm = LLM(
            api_type=_resolve(settings.WRITER_API_TYPE, settings.LLM_DEFAULT_API_TYPE),
            api_key=_resolve(settings.WRITER_API_KEY, settings.LLM_DEFAULT_API_KEY),
            model=_resolve(settings.WRITER_MODEL, settings.LLM_DEFAULT_MODEL),
            base_url=_resolve(settings.WRITER_BASE_URL, settings.LLM_DEFAULT_BASE_URL),
            task_id=self.task_id,
            max_tokens=_resolve(settings.WRITER_MAX_TOKENS, settings.LLM_DEFAULT_MAX_TOKENS),
        )

        return coordinator_llm, modeler_llm, coder_llm, writer_llm
