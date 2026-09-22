from core.core_data import LocalData


def collect(data: LocalData) -> str:
    return "Timestamp: " + data.global_data.datetime()
