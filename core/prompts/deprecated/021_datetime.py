from core.core_data import LocalData
from core.helpers import xml_tag


def collect(data: LocalData) -> str:
    return xml_tag(
        data.global_data.datetime(),
        "datetime",
        "Current date and time can be useful for time-aware responses.",
    )
