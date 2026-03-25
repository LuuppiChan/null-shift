import asyncio
import logging
from typing import Any, Dict

from .classifier import DifficultyClassifier, TaskType
from .bus import WrapperBus
from .config import AgentConfig

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self, bus: WrapperBus, config: AgentConfig):
        self.bus = bus
        self.config = config
        
        # Inbound tasks pile up here until the Core is free
        self.task_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        
        # Internal queues for active tracking
        self._sub_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._signal_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        
    async def start(self):
        """Starts all background routers and the main consumer loop."""
        asyncio.create_task(self._route_ui(), name="route_ui")
        asyncio.create_task(self._route_sub(), name="route_sub")
        asyncio.create_task(self._route_signals(), name="route_signals")
        
        await self._consumer_loop()

    async def _route_ui(self):
        async for payload in self.bus.listen_ui():
            logger.info("New UI task received. Queuing.")
            await self.task_queue.put(payload)

    async def _route_sub(self):
        async for topic, payload in self.bus.listen_core_sub():
            if topic == "assistant.stream.done":
                await self._sub_queue.put(payload)

    async def _route_signals(self):
        async for payload in self.bus.listen_signals():
            await self._signal_queue.put(payload)

    async def _clear_queues(self):
        """Drain stale messages between tasks to prevent desync."""
        while not self._sub_queue.empty():
            self._sub_queue.get_nowait()
        while not self._signal_queue.empty():
            self._signal_queue.get_nowait()

    async def _consumer_loop(self):
        logger.info("Orchestrator ready. Waiting for tasks...")
        while True:
            task_payload = await self.task_queue.get()
            try:
                task_type = await DifficultyClassifier.determine(task_payload)
                logger.info(f"Processing task assigned difficulty: {task_type}")
                
                await self._clear_queues()
                
                body = task_payload.get("body", "")
                
                # Pre-process agentic tasks with their respective templates
                if task_type == TaskType.AUTONOMOUS_STRICT:
                    body = self.config.prompts_strict_template.format(body=body)
                elif task_type == TaskType.AUTONOMOUS_TRAJECTORY:
                    body = self.config.prompts_trajectory_template.format(body=body)
                
                # Format to instant
                forward_payload = {
                     "body": body,
                     "title": task_payload.get("title", ""),
                }
                if "tools" in task_payload:
                    forward_payload["tools"] = task_payload["tools"]

                await self.bus.send_to_core("input.instant", forward_payload)

                if task_type in (TaskType.SIMPLE, TaskType.TOOL_ASSISTED):
                    await self._handle_simple_task()
                elif task_type in (TaskType.AUTONOMOUS_STRICT, TaskType.AUTONOMOUS_TRAJECTORY):
                    await self._handle_autonomous_task()
                
            except Exception as e:
                logger.error(f"Error executing task: {e}", exc_info=True)
            finally:
                self.task_queue.task_done()
                logger.info("Task completed. Ready for next.")

    async def _handle_simple_task(self):
        """Wait for the exact first stream.done indicating vector finish."""
        logger.debug("Waiting for core finish...")
        await self._sub_queue.get()
        logger.info("Core finished simple task.")

    async def _handle_autonomous_task(self):
        """Run the autonomous loop tracking native agent signals."""
        logger.info("Entering Autonomous Loop.")
        while True:
            logger.debug("Waiting for vector stream finish...")
            
            # The LLM finishes its stream iteration
            await self._sub_queue.get()
            logger.debug("Vector finished a turn. Checking for Intercepts...")
            
            try:
                # If a Native Tool was called during that turn, it would have sent a signal just before finishing.
                # We wait briefly to pop it.
                signal = await asyncio.wait_for(self._signal_queue.get(), timeout=1.0)
                signal_type = signal.get("signal")
                
                logger.info(f"Intercepted Native Signal: {signal_type}")
                
                if signal_type == "task_complete":
                    return # Breaks autonomous loop safely
                    
                bounce_msg = None
                if signal_type == "step_done":
                    bounce_msg = self.config.prompts_step_done
                elif signal_type in ("plan_updated", "step_reverted"):
                    bounce_msg = self.config.prompts_plan_updated
                elif signal_type in ("plan_created", "research_done"):
                    bounce_msg = self.config.prompts_phase_done
                else:
                    logger.warning(f"Unknown or unhandled tool signal: {signal_type}")

                if bounce_msg:
                    await self.bus.send_to_core("input.instant", {
                        "body": bounce_msg,
                        "title": self.config.prompts_agentic_loop_title
                    })
                else:
                     logger.warning("No bounce message matched. Exiting autonomous loop to prevent stall.")
                     return

            except asyncio.TimeoutError:
                # If no signal arrived, the LLM replied natively without hitting a loop tool.
                # We bounce it back to wake it up and force it to use a tool to end/proceed.
                logger.warning("No Native Tool Signal arrived after Turn! Bouncing to wake up.")
                await self.bus.send_to_core("input.instant", {
                    "body": self.config.prompts_wake_up,
                    "title": self.config.prompts_agentic_loop_title
                })
