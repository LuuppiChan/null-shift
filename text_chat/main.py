import logging
import readline
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import sleep
from typing import cast
import signal

import zmq
from pydantic import BaseModel
from termcolor import colored

from global_tools import ConfigManager
from global_types import BusMessage, Commands, Difficulty, InputMessage, MessageTopic

logger = logging.getLogger(__name__)
logging.basicConfig()

ctx = zmq.Context()


class ChatConfig(BaseModel):
    input_url: str = "tcp://127.0.0.1:5555"
    output_url: str = "tcp://127.0.0.1:5556"


manager = ConfigManager(Path("./config.toml"), ChatConfig())


class TextChat:
    def __init__(self) -> None:
        cfg = manager.get_config()
        self.core_sub = ctx.socket(zmq.SUB)
        self.core_push = ctx.socket(zmq.PUSH)
        self.shutdown = Event()

        self.core_sub.connect(cfg.output_url)
        self.core_push.connect(cfg.input_url)
        self.core_sub.subscribe(b"")

    def listen_core(self):
        last_was_reason = False
        while True:
            frames = self.core_sub.recv_multipart()
            if self.shutdown.is_set():
                return

            message: BusMessage | None = BusMessage.decoded(frames)
            if message is None:
                logger.warning("Error parsing message.")
                continue

            match message.topic:
                case MessageTopic.STREAM:
                    reasoning = message.payload.get("reasoning", "")
                    text = message.payload.get("text", "")

                    if last_was_reason and text:
                        print()

                    print(colored(reasoning, "dark_grey"), end="", flush=True)
                    print(colored(text, "light_cyan"), end="", flush=True)

                    last_was_reason = bool(reasoning)
                case MessageTopic.TOOL_CALL:
                    tool_name = message.payload.get("tool_name", "No name")
                    args = "\n".join([f"    {k}={v}" for k, v in message.payload.get("args", {}).items()])
                    print(colored(f"\nTOOL: [{tool_name}] called\n{args}", "dark_grey"))
                    # print("\n".join([f"   {k}: {v}" for k, v in message.payload.get("args", {})]))
                    code = message.payload.get("args", {}).get("code", "")
                    if code:
                        print(colored(code, "green"))
                case MessageTopic.TOOL_RESULT:
                    tool_name = message.payload.get("tool_name", "No name")
                    print(colored(f"TOOL: [{tool_name}] returned", "dark_grey"))
                    content = message.payload.get("content", "")
                    if content:
                        print(colored(content, "yellow"))
                case MessageTopic.FINISHED:
                    print(colored("\n> ", "light_red"), end="", flush=True)
                case MessageTopic.STARTED:
                    print("\n")
                case MessageTopic.ABORT:
                    print("\nStream aborted\n")

    def listen_input(self):
        msg: BusMessage | None = None
        difficulty: Difficulty = Difficulty.TOOL_ASSISTED
        title: str | None = None
        goal: str | None = None

        while True:
            text = input(colored("> ", "light_red"))

            if self.shutdown.is_set():
                return

            if text.startswith("/"):
                parts = text.split(" ", 1)
                if len(parts) == 1:
                    command = parts[0]
                    arg = ""
                else:
                    command, arg = parts

                match command:
                    case "/abort" | "/a":
                        msg = BusMessage(
                            topic=MessageTopic.COMMAND,
                            payload={"command": Commands.ABORT},
                        )
                    case "/difficulty" | "/d":
                        match arg.lower():
                            case "at" | "t":
                                arg = Difficulty.AUTONOMOUS_TRAJECTORY
                            case "as":
                                arg = Difficulty.AUTONOMOUS_STRICT
                            case "s":
                                arg = Difficulty.SIMPLE
                            case "ta":
                                arg = Difficulty.TOOL_ASSISTED
                        if arg in Difficulty:
                            difficulty = cast(Difficulty, arg)
                            print(f"{difficulty=}")
                        else:
                            print(
                                f"Invalid difficulty {arg}. Available: {list(Difficulty)}"
                            )
                        msg = None
                    case "/title" | "/t":
                        title = arg
                        msg = None
                    case "/goal" | "/g":
                        goal = arg
                        msg = None
                    case "/q" | "/quit":
                        signal.raise_signal(signal.SIGINT)
                        break
                    case "/" | "/help" | _:
                        print(
                            "Commands: '/abort' (/a), '/difficulty' (/d), '/title' (/t), '/goal' (/g)"
                        )
                        msg = None
            elif text.startswith("#"):
                continue
            elif not text:
                continue
            else:
                msg = BusMessage(
                    topic=MessageTopic.INPUT,
                    payload=InputMessage(
                        title=title, body=text, difficulty=difficulty, goal=goal
                    ).model_dump(),
                )
                title = None
                goal = None

            if msg:
                print(f"Sending {msg.model_dump()}")
                self.core_push.send_multipart(msg.encoded())


def main():
    chat = TextChat()
    with ThreadPoolExecutor(3) as pool:
        try:
            input = pool.submit(chat.listen_input)
            core = pool.submit(chat.listen_core)

            while input.running() or core.running():
                sleep(0.1)

        except KeyboardInterrupt:
            print("Shutdown event set.")
            chat.shutdown.set()
        finally:
            chat.core_push.close()
            chat.core_sub.close()
            ctx.destroy()
