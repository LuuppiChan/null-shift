from core.core_data import LocalData
from core.helpers import xml_tag


def collect(data: LocalData) -> str:
    return xml_tag(
        data.global_data.scratchpad(),
        "assistant_scratchpad_path",
        "This is your dedicated workspace. You can freely read, write, and modify files here. Use this directory to draft code, store intermediate data, write down step-by-step plans, or keep track of your thoughts during complex tasks. Consider it your working memory.",
    )
