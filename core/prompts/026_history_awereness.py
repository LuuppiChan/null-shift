from core.config import manager
from core.helpers import xml_tag


def collect() -> str:
    cfg = manager.get_config()
    parts = []
    # static
    parts.append(xml_tag("Be aware of history trimming which cuts old parts of history away.\nIf you need to note down a detail do it immediately after noticing it.\nRemember that you can always update a file later.", "history_trimming"))

    if cfg.history.compression:
        parts.append(xml_tag("History compression is enabled. There may be a summary of past events at the beginning of this conversation.", "history_compression"))
    return "\n\n".join(parts)
