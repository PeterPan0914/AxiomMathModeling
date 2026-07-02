from app.core.prompts.coordinator import COORDINATOR_PROMPT, FORMAT_QUESTIONS_PROMPT, get_coordinator_system_prompt
from app.core.prompts.modeler import MODELER_PROMPT, get_modeler_system_prompt, get_counterfactual_prompt
from app.core.prompts.coder import CODER_PROMPT
from app.core.prompts.writer import (
    get_writer_prompt,
    get_writer_system_prompt,
    CHAPTER_ABSTRACT,
    CHAPTER_PROBLEM_ANALYSIS,
    CHAPTER_MODEL,
    CHAPTER_RESULTS,
    CHAPTER_ROBUSTNESS,
    CHAPTER_EVALUATION,
    CHAPTER_DEFAULT,
)
from app.core.prompts.reviewer import get_reviewer_prompt
from app.core.prompts.literature import get_literature_research_prompt
from app.core.prompts.outline import get_outline_prompt
from app.core.prompts.consistency import get_consistency_check_prompt
from app.core.prompts.shared import get_reflection_prompt, get_completion_check_prompt
from app.core.prompts.prompt_engineering import (
    get_chain_of_thought_prompt,
    get_self_consistency_prompt,
    get_tree_of_thought_prompt,
    get_reflexion_prompt,
    get_academic_writing_prompt,
    get_sensitivity_analysis_prompt,
    get_model_validation_prompt,
    get_modeler_cot_prompt,
    get_writer_cot_prompt,
    get_coder_cot_prompt,
)

__all__ = [
    "COORDINATOR_PROMPT",
    "FORMAT_QUESTIONS_PROMPT",
    "get_coordinator_system_prompt",
    "MODELER_PROMPT",
    "get_modeler_system_prompt",
    "get_counterfactual_prompt",
    "CODER_PROMPT",
    "get_writer_prompt",
    "get_writer_system_prompt",
    "CHAPTER_ABSTRACT",
    "CHAPTER_PROBLEM_ANALYSIS",
    "CHAPTER_MODEL",
    "CHAPTER_RESULTS",
    "CHAPTER_ROBUSTNESS",
    "CHAPTER_EVALUATION",
    "CHAPTER_DEFAULT",
    "get_reviewer_prompt",
    "get_literature_research_prompt",
    "get_outline_prompt",
    "get_consistency_check_prompt",
    "get_reflection_prompt",
    "get_completion_check_prompt",
    "get_chain_of_thought_prompt",
    "get_self_consistency_prompt",
    "get_tree_of_thought_prompt",
    "get_reflexion_prompt",
    "get_academic_writing_prompt",
    "get_sensitivity_analysis_prompt",
    "get_model_validation_prompt",
    "get_modeler_cot_prompt",
    "get_writer_cot_prompt",
    "get_coder_cot_prompt",
]
