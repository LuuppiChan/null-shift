import asyncio
import json
import logging
import zmq
import zmq.asyncio

async def main():
    logging.basicConfig(level=logging.INFO)
    ctx = zmq.asyncio.Context()
    
    # Setup PUSH socket to send commands/inputs
    push = ctx.socket(zmq.PUSH)
    push.connect("tcp://127.0.0.1:5555")
    
    # Setup SUB socket to listen to replies and stream
    sub = ctx.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5556")
    sub.subscribe("")  # Subscribe to all topics for testing
    
    print("--- Testing State Polling ---")
    await push.send_multipart([
        b"input.command",
        json.dumps({
            "cmd": "poll_state",
            "reply_topic": "test.state_reply"
        }).encode("utf-8")
    ])
    
    while True:
        topic_bytes, payload_bytes = await sub.recv_multipart()
        topic = topic_bytes.decode()
        payload = json.loads(payload_bytes)
        if topic == "test.state_reply":
            print(f"[{topic}] {payload}")
            print("=> Polling successful!\n")
            break

    print("--- Testing External Tool Injection ---")
    mock_tool = {
        "type": "function",
        "function": {
            "name": "get_time_in_zone",
            "description": "Get the current time in a specified timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string"}
                },
                "required": ["timezone"]
            }
        }
    }
    
    await push.send_multipart([
        b"input.instant",
        json.dumps({
            "body": "What time is it in America/Los_Angeles?",
            "tools": [mock_tool]
        }).encode("utf-8")
    ])
    
    while True:
        topic_bytes, payload_bytes = await sub.recv_multipart()
        topic = topic_bytes.decode()
        payload = json.loads(payload_bytes)
        
        if topic == "action.request":
            print(f"[{topic}] {payload}")
            if payload.get("tool") == "get_time_in_zone":
                print(f"=> Action request intercepted successfully: {payload['args']}")
                # Send mock result mimicking an Action node
                await push.send_multipart([
                    b"action.result",
                    json.dumps({
                        "call_id": payload["call_id"],
                        "result": "It is 08:00 AM."
                    }).encode("utf-8")
                ])
                print("=> Sent mock action.result back to Cognition\n")
                
        if topic == "assistant.stream.done":
            print(f"[{topic}] {payload}")
            print(f"=> Turn finished. Reason: {payload.get('reason')}")
            print(f"=> Used tools: {payload.get('tool_calls')}")
            break

    print("\n--- All Tests Done ---")
    push.close()
    sub.close()
    ctx.term()

if __name__ == "__main__":
    asyncio.run(main())
