import asyncio
import logging
from .config import load_config
from .bus import WrapperBus
from .agent import AgentOrchestrator

def main():
    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] agent-wrapper - %(message)s"
    )
    
    bus = WrapperBus(config)
    agent = AgentOrchestrator(bus, config)
    
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        asyncio.run(bus.close())

if __name__ == "__main__":
    main()
