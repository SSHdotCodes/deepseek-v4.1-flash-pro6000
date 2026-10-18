# Benchmark prompt sources and licenses

The prompts are stored as token IDs of DeepSeek-V4.1-Flash-1017's official encoding.

- `chat-000.json`, `chat-001.json`, `chat-002.json`: written for this project; MIT, like the rest of the repository.
- `mt-000.json`: derived from [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), shard `train-00000-of-00014`, conversation `004cac9e7e4cc15306fca3c32f5647db`, with the trailing assistant turn dropped. WildChat-1M is made available under the Open Data Commons Attribution License (ODC-By) v1.0, [`LICENSES/ODC-By-1.0.txt`](../../LICENSES/ODC-By-1.0.txt). Cite: Zhao et al., *WildChat: 1M ChatGPT Interaction Logs in the Wild*, ICLR 2024.
- `agentic-000.json`, `agentic-001.json`: derived from [ProCreations/betterwright-agent-traces](https://huggingface.co/datasets/ProCreations/betterwright-agent-traces) (accepted split), DeepSeek-V4.1-Flash browser-agent runs replayed up to a turn boundary. That dataset is licensed under the Apache License 2.0, [`LICENSES/Apache-2.0.txt`](../../LICENSES/Apache-2.0.txt). Changes: conversations truncated at a turn boundary, tool calls normalized to the OpenAI format, and the result encoded to token IDs.
