# Gemma 3 Tool Parser for vLLM

This folder contains a custom tool parser and chat template designed to enable robust tool calling (function calling) for Gemma 3 models in [vLLM](https://github.com/vllm-project/vllm).

## Overview

Gemma 3 models use Python-like syntax for calling tools inside ```tool_code``` blocks. This project provides:
1. **`alphadeep_gemma3_tool_parser.py`**: A custom parser class implementing vLLM's `ToolParser` interface to extract, parse, and stream tool calls from model outputs.
2. **`tool_chat_template_gemma3_json.jinja`**: A customized Jinja chat template that formats tool declarations as Python function stubs inside the system/instruction context, guiding the model to emit tool calls correctly.

This implementation is based on the official vLLM Gemma 3 tool calling example and was built with the help of Gemini CLI.

## Verified Compatible Models

The parser and template have been verified to be compatible with:
* **Gemma 3 27B IT** (`google/gemma-3-27b-it`)
* **Gemma 3 12B IT** (`google/gemma-3-12b-it`)

## Heuristic Fixes for Tool Calling

Standard Python AST parsing of tool blocks can fail if the model makes minor formatting mistakes. This parser implements several robust heuristic fixes prior to AST parsing to maximize reliability:

1. **Missing Comma Insertion**: Corrects instances where the model emits arguments separated only by spaces (e.g., `func(a="foo" b=42)` is fixed to `func(a="foo", b=42)`).
2. **Lowercase/JSON-style Booleans and Nulls**: Replaces JSON-like parameters `=true`, `=false`, and `=null` with valid Python literals `=True`, `=False`, and `=None` so that the Python AST parser can read them.
3. **Stray/Double Commas Cleanup**: Normalizes consecutive or excess commas (such as `,,` or `, ,`) that may result from replacing booleans or other format adjustments.

## How It Works

1. **Prompt Template**: When tools are provided to the model, they are formatted as standard Python function signatures with docstrings.
2. **Output Block**: The model responds by wrapping function calls inside a ````tool_code ... ```` markdown block.
3. **Parsing**: The parser extracts the content of the ````tool_code```` block, applies the heuristic fixes, parses the syntax using Python's `ast` module, and maps the calls back to the OpenAI tool call protocol structure used by vLLM (both for standard and streaming APIs).

---

## Acknowledgements

This project was built with the support of the [Google TPU Research Cloud (TRC)](https://sites.research.google/trc/).

<p align="left">
  <img src="logotrc.png" width="200" alt="Google TPU Research Cloud Logo" />
</p>

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
