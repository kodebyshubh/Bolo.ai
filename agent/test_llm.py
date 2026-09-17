"""
Sanity-check agent/llm.py: direct-reply and tool-call paths, including a
Hinglish tool-call input and a nonexistent order id.

Usage:
    python agent/test_llm.py
"""
import os
import sys

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.llm import respond

load_dotenv()

TEST_INPUTS = [
    "What's the best way to store fresh basil?",
    "What's the status of order 12345?",
    "mera order 67890 kab aayega",
    "kaha hai mera order 99999",
]


def main():
    for text in TEST_INPUTS:
        print(f"> {text}")
        print(respond(text))
        print()


if __name__ == "__main__":
    main()
