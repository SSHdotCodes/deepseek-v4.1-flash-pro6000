"""Machine-checkable summary of the backend contract types.

Run: python -m ds41_api.contract_check
Prints the canonical JSON wire shapes so engine teams can diff against their
implementation without reading the Python.
"""

import json

from .backend import FinishEvent, SamplingParams, TokenEvent, Usage


def main() -> None:
    print(json.dumps(
        {
            "sampling": SamplingParams().to_dict(),
            "token_event": TokenEvent(token_id=1, logprob=-0.5, top_logprobs=[
                {"token_id": 1, "logprob": -0.5},
                {"token_id": 2, "logprob": -1.5},
            ]).to_dict(),
            "finish_event": FinishEvent(
                finish_reason="stop", usage=Usage(prompt_tokens=10, completion_tokens=3)
            ).to_dict(),
            "ops": {
                "submit": {
                    "op": "submit",
                    "request_id": "<uuid4>",
                    "token_ids": [0, 1, 2],
                    "sampling": SamplingParams().to_dict(),
                    "images": [{"data": "<base64 or null>", "url": "<str or null>"}],
                    "conversation_id": "<str or null>",
                },
                "cancel": {"op": "cancel", "request_id": "<uuid4>"},
            },
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
