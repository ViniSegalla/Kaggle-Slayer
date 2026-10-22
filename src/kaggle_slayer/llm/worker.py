"""Isolated provider transport. No shared run/store access or generated code."""

import json
import sys

from kaggle_slayer.config import Settings
from kaggle_slayer.llm.provider import ActionLLM


def main():
    request = json.load(sys.stdin)
    response, diagnostic = ActionLLM(Settings(_env_file=None, **request["settings"])).decide(
        request["role"],
        request["context"],
        timeout=request["timeout"],
        max_tokens=request["max_tokens"],
    )
    print(
        json.dumps(
            {
                "response": response.model_dump(mode="json") if response else None,
                "diagnostic": diagnostic,
            }
        )
    )


if __name__ == "__main__":
    main()
