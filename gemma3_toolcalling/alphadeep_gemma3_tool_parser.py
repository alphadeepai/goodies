import ast
import json
import regex as re
from collections.abc import Sequence

from typing import Any, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class _UnexpectedAstError(Exception):
    pass


class Gemma3ToolParser(ToolParser):
    """
    Parser for Gemma 3 using the ```tool_code ... ``` format.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.last_streamed_text_index = 0
        self.in_tool_code = False
        self.tool_code_buffer = ""

    def _split_text_and_tools(self, model_output: str) -> tuple[str, str]:
        """
        Split model output into text content and tool calls section.
        """
        start_marker = "```tool_code\n"
        end_marker = "\n```"

        start_idx = model_output.find(start_marker)
        if start_idx != -1:
            text_part = model_output[:start_idx].strip()
            tools_part = model_output[start_idx + len(start_marker) :]

            end_idx = tools_part.find(end_marker)
            if end_idx != -1:
                tools_part = tools_part[:end_idx]
            else:
                # Still strip any ending backticks if generation stopped early
                tools_part = tools_part.rstrip("`\n")

            return text_part, tools_part

        return model_output, ""

    def _parse_tool_code(self, tools_section: str) -> list[ast.Call]:
        # Heuristic fix for missing commas between arguments
        tools_section = re.sub(
            r'([\]"\'\d])\s*([a-zA-Z_]\w*=)', r"\1, \2", tools_section
        )

        # Heuristic fix for lowercase boolean params
        tools_section = tools_section.replace("=false", "=False").replace("=true", "=True").replace("=null", "=None")


        # Heuristic fix for multi comma and last comma
        tools_section = tools_section.replace("=False", "=False,").replace("=True", "=True,").replace("=None", "=None,")
        while ",," in tools_section or ", ," in tools_section:
            tools_section = tools_section.replace(",,", ",").replace(", ,", ",")


        # ast.parse parses multiple statements into a module.body list of ast.Expr
        module = ast.parse(tools_section)
        calls = []
        for stmt in module.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                calls.append(stmt.value)
            else:
                dump = ast.dump(module, indent=4)
                raise _UnexpectedAstError(f"Tool output must be function calls {stmt} {stmt.__dict__} {tools_section=} {dump=}")
        return calls

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:

        text_content, tools_section = self._split_text_and_tools(model_output)

        if not tools_section:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            calls = self._parse_tool_code(tools_section)
            tool_calls = [self._handle_single_tool(c) for c in calls]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=text_content if text_content else None,
            )
        except Exception:
            logger.exception("Error extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:

        # Simplified streaming that buffers tool_code block until the end of generation
        # and parses it fully. For text outside tool_code, streams it normally.

        start_marker = "```tool_code\n"
        start_idx = current_text.find(start_marker)

        # We haven't found a tool call start marker yet
        if start_idx == -1:
            # If the current text ends with parts of the start marker, withhold it
            # so we don't stream "```" out to the user if it's about to be a tool block
            for i in range(1, len(start_marker) + 1):
                if current_text.endswith(start_marker[:i]):
                    return None

            # Not withheld, stream normally
            start_index = self.last_streamed_text_index
            self.last_streamed_text_index = len(current_text)
            if start_index < len(current_text):
                return DeltaMessage(content=current_text[start_index:])
            return None


        # Ensure we have streamed all the text before the tool call
        if self.last_streamed_text_index < start_idx:
            to_stream = current_text[self.last_streamed_text_index : start_idx]
            self.last_streamed_text_index = start_idx
            if to_stream:
                return DeltaMessage(content=to_stream)

        # Check if the block has closed
        end_marker = "\n```"
        tools_part = current_text[start_idx + len(start_marker) :]
        end_idx = tools_part.find(end_marker)

        # If it hasn't closed, we suppress the stream output
        if end_idx == -1:
            return None

        # It has closed, parse and stream the tools exactly once
        if not self.in_tool_code:
            self.in_tool_code = True
            tools_section = tools_part[:end_idx].strip()

            try:
                calls = self._parse_tool_code(tools_section)
                tool_calls = []
                tool_deltas = []

                for index, e in enumerate(calls):
                    call = self._handle_single_tool(e)
                    tool_calls.append(call)

                    delta = DeltaToolCall(
                        id=call.id,
                        type="function",
                        index=index,
                        function=DeltaFunctionCall(
                            name=call.function.name,
                            arguments=call.function.arguments,
                        ),
                    )
                    tool_deltas.append(delta)

                self.prev_tool_call_arr = [{"arguments": {}} for _ in tool_deltas]
                return DeltaMessage(tool_calls=tool_deltas)
            except Exception as e:
                logger.exception(f"Streaming parse error inside tool_code block. {e=} {tools_section=}")

        # If we already sent the tools and there's trailing text we pass it through.
        # Typically the generation ends after the tool output block, but gemma3 is built different.
        return DeltaMessage(content=delta_text)

    def _get_parameter_value(self, val: ast.expr) -> Any:
        if isinstance(val, ast.Constant):
            return val.value
        elif isinstance(val, ast.Dict):
            result = {}
            for k, v in zip(val.keys, val.values):
                if isinstance(k, ast.Constant):
                    result[k.value] = self._get_parameter_value(v)
            return result
        elif isinstance(val, ast.List):
            return [self._get_parameter_value(v) for v in val.elts]
        else:
            dump = ast.dump(val, indent=4)
            raise _UnexpectedAstError(f"Tool call arguments must be literals {val=} {val.__dict__=} {dump=}")

    def _handle_single_tool(self, call: ast.Call) -> ToolCall:
        if not isinstance(call.func, ast.Name):
            dump = ast.dump(call, indent=4)
            raise _UnexpectedAstError(f"Invalid tool call name {call=} {call.__dict__=} {dump=}")
        function_name = call.func.id
        arguments = {}
        for keyword in call.keywords:
            if keyword.arg is not None:
                arguments[keyword.arg] = self._get_parameter_value(keyword.value)
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name, arguments=json.dumps(arguments, ensure_ascii=False)
            ),
        )
