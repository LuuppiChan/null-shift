import asyncio
import importlib

async def test_stt():
    import transcribe
    node = transcribe.STTNode(asyncio.get_running_loop())
    print("MOCK TEST", node)
    # The actual tests should just listen to ZMQ
    
if __name__ == "__main__":
    asyncio.run(test_stt())
