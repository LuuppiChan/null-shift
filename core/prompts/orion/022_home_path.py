from core.core_data import LocalData
from core.helpers import xml_tag


def collect(data: LocalData) -> str:
    return xml_tag(data.global_data.home_path(), "user_home_path")
