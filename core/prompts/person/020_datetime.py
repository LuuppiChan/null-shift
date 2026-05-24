from core.core_data import LocalData


def collect(data: LocalData) -> str:
    return f'<datetime description="Current date and time can be useful for time-aware responses.">\n{data.global_data.datetime()}\n</datetime>'
