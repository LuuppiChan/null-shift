from enum import StrEnum
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

from global_types import BusMessage


class Action(StrEnum):
    RETURN = "return"
    DOM = "dom"
    CLICK = "click"
    TYPE = "type"
    EXTRACT_ELEMENT = "extract_element"
    PRESS_KEY = "press_key"
    SCROLL = "scroll"
    HOVER = "hover"
    MISC_ACTION = "misc_action"
    LIST_TABS = "list_tabs"
    SWITCH_TAB = "switch_tab"
    CLOSE_TAB = "close_tab"
    NEW_TAB = "new_tab"
    NAVIGATE = "navigate"
    SET_SLIDER = "set_slider"
    PAGE_SCREENSHOT = "page_screenshot"
    ELEMENT_SCREENSHOT = "element_screenshot"
    SELECT_OPTION = "select_option"
    VIDEO_CONTROL = "video_control"


class BrowserMessage(BaseModel):
    action: Action
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_bus_msg(cls, msg: BusMessage) -> Optional["BrowserMessage"]:
        if msg.topic in Action:
            return cls(action=cast(Action, msg.topic), kwargs=msg.payload)
        return None
