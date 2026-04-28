import argparse


def launch_tools():
    import multiprocessing

    from tools.browser.__main__ import run as run_browser

    print("Starting tool processes...")
    processes = [multiprocessing.Process(target=run_browser, name="BrowserTool")]

    for p in processes:
        p.start()
        print(f"Started {p.name} (PID: {p.pid})")

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nShutting down tool processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
        print("All tools shut down gracefully.")


def main():
    parser = argparse.ArgumentParser(
        description="Null-shift: next generation modular AI agent system."
    )
    parser.add_argument("module", help="What module to launch")

    # Optional argument (with a default value)
    # parser.add_argument("-g", "--greeting", default="Hello", help="The greeting to use")

    args = parser.parse_args()

    match args.module:
        case "core":
            import core

            core.run()
        case "text_debug":
            print("Todo")
        case "text":
            import text_chat
        case "tools":
            launch_tools()
        case _:
            print(f"Unknown module: {args.module}")


if __name__ == "__main__":
    main()
