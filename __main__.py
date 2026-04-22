import argparse


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
        case _:
            print(f"Unknown module: {args.module}")


if __name__ == "__main__":
    main()
