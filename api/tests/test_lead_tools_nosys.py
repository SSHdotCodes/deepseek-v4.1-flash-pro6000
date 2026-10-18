"""Lead decision 2026-10-17: tools without a system message are rendered on
/v1/chat/completions too (deepseek-recipe 0.1.1 behaves the same way)."""
from pathlib import Path

import os
import pytest

from ds41_api.encoder import Encoder
from ds41_api.schemas import ChatCompletionRequest

TOK = Path(os.environ.get("DS41_TOKENIZER", Path(__file__).resolve().parents[2] / "models/DeepSeek-V4.1-Flash-1017/tokenizer.json"))
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


@pytest.mark.skipif(not TOK.exists(), reason="tokenizer.json not present")
def test_tools_without_system_are_rendered_and_match_empty_system():
    enc = Encoder.from_file(str(TOK))
    nosys = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "Weather in Paris?"}], tools=TOOLS)
    withsys = ChatCompletionRequest(model="m", messages=[{"role": "system", "content": ""},
                                                         {"role": "user", "content": "Weather in Paris?"}], tools=TOOLS)
    p1, _ = enc.build_prompt(nosys)
    p2, _ = enc.build_prompt(withsys)
    assert "## Tools" in p1 and "get_weather" in p1
    assert p1 == p2


@pytest.mark.skipif(not TOK.exists(), reason="tokenizer.json not present")
def test_tool_choice_none_still_omits_tools():
    enc = Encoder.from_file(str(TOK))
    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], tools=TOOLS, tool_choice="none")
    p, _ = enc.build_prompt(req)
    assert "## Tools" not in p
