import sys
import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Mock logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Add current directory to sys.path
sys.path.append(os.getcwd())

from registry import tool_registry
from config import manager

async def test_tool_lifecycle():
    print("=== STARTING TOOL LIFECYCLE TEST ===")
    
    # --- Part 1: Initial Load ---
    print("\n[1/3] Initial tool load...")
    try:
        tools = await tool_registry.get_tools()
        print(f"Loaded {len(tools)} tools initially.")
    except Exception as e:
        print(f"FAILED initial load: {e}")
        return

    # --- Part 2: Pickling Verification (Multiprocessing) ---
    # We force a refresh by setting the debounce to 0.
    # This will trigger the multiprocessing.Process which pickles the load function.
    manager.get_config().core_tools_min_refresh_delay = 0
    
    print("\n[2/3] Second load (verifying pickling with cached tools)...")
    try:
        # We need to wait a tiny bit to ensure the timestamp check passes if it used the same second
        await asyncio.sleep(0.1)
        tools = await tool_registry.get_tools()
        print(f"SUCCESS: Second load completed with {len(tools)} tools.")
    except Exception as e:
        print(f"CRITICAL ERROR (Pickling/IPC issue): {e}")

    # --- Part 3: Debounce Logic Verification ---
    # We set a long delay and check if calling it repeatedly pushes the window.
    delay = 60
    manager.get_config().core_tools_min_refresh_delay = delay
    tool_registry.refreshed = datetime.now()
    
    print(f"\n[3/3] Debounce logic check (delay={delay}s)...")
    original_refreshed = tool_registry.refreshed
    
    await asyncio.sleep(1)
    await tool_registry.get_tools()
    
    new_refreshed = tool_registry.refreshed
    if new_refreshed > original_refreshed:
        print("BUG FOUND: Debounce timestamp was updated during a 'fresh enough' check.")
        print(f"Original: {original_refreshed}")
        print(f"New:      {new_refreshed}")
        print("This will prevent tools from EVER refreshing if called more frequently than the delay.")
    else:
        print("SUCCESS: Debounce logic correctly maintained the original timestamp.")

    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test_tool_lifecycle())
