"""ds41_api: OpenAI-compatible serving front-end for DeepSeek-V4.1-Flash-1017.

Layers:
  encoder.py    - request -> encoded prompt string -> token ids (reference encoding)
  parser.py     - completion text -> message; incremental stream state machine
  backend.py    - Backend ABC + event types (engine contract)
  fake_backend.py   - scripted in-process backend for tests
  socket_backend.py - Unix-socket JSON-lines backend client/server
  server.py     - FastAPI app: /v1/models, /v1/chat/completions
"""

__version__ = "0.1.0"
