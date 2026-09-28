import logging
from pathlib import Path

from dotenv import load_dotenv

from src.agent import DocumentAgent


def configure_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "agent.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    load_dotenv()
    configure_logging()

    try:
        agent = DocumentAgent("data/sample_document.txt")
    except Exception as exc:
        print(f"Startup error: {exc}")
        return

    print("\nPointStar Part 3 - Gemini Document Q&A Agent")
    print("Commands: 'reset' clears memory, 'exit' quits.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Goodbye!")
            break

        if user_input.lower() == "reset":
            agent.reset_memory()
            print("Agent: Conversation memory has been reset.\n")
            continue

        try:
            answer = agent.ask(user_input)
            print(f"Agent: {answer}\n")
        except Exception as exc:
            logging.getLogger("pointstar_agent").exception("Agent request failed.")
            print(f"Agent error: {exc}\n")


if __name__ == "__main__":
    main()
