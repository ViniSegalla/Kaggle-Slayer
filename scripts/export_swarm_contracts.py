"""Export the executable v1 wire schemas and role prompts for endpoint integration."""

import argparse
from pathlib import Path

from kaggle_slayer.agents.swarm import ROLE_PROMPTS, role_prompt, role_response_schema
from kaggle_slayer.schemas.actions import VERSION, AgentResponse
from kaggle_slayer.storage.store import write_json
from kaggle_slayer.tools.swarm_tools import BOUNDS, CAPABILITIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/contracts") / VERSION)
    args = parser.parse_args()
    write_json(args.output / "actions.schema.json", AgentResponse.model_json_schema())
    write_json(
        args.output / "tools.schema.json",
        {
            "version": VERSION,
            "tools": {
                name: {
                    "roles": cap.roles,
                    "experiment": cap.experiment,
                    "description": cap.description,
                    "arguments": cap.arguments.model_json_schema(),
                }
                for name, cap in CAPABILITIES.items()
            },
            "parameter_bounds": BOUNDS,
        },
    )
    for role in ROLE_PROMPTS:
        write_json(args.output / f"{role}.actions.schema.json", role_response_schema(role))
        (args.output / f"{role}.txt").write_text(role_prompt(role), encoding="utf-8")


if __name__ == "__main__":
    main()
