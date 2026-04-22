import asyncio
import zmq.asyncio
import json

async def mock_core():
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind("tcp://127.0.0.1:5555")
    print("Mock core listener started on tcp://127.0.0.1:5555")
    
    try:
        while True:
            frames = await sock.recv_multipart()
            topic = frames[0].decode()
            payload = json.loads(frames[1])
            print(f"RECEIVED! Topic: {topic}, Payload: {payload}")
    except asyncio.CancelledError:
        pass
    finally:
        sock.close()

if __name__ == "__main__":
    try:
        asyncio.run(mock_core())
    except KeyboardInterrupt:
        pass
