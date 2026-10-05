import json
import uuid
from typing import Any, Sequence

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool


class OllamaGenerateChatModel(BaseChatModel):
    api_url: str
    model: str
    proxy_url: str | None = None
    timeout: float = 180

    @property
    def _llm_type(self) -> str:
        return "ollama-fastapi-generate"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"api_url": self.api_url, "model": self.model}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        tool_schemas = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=tool_schemas, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools", [])
        prompt = self._render_prompt(messages, tools)
        response_text = self._request(prompt)
        allowed_tools = {
            tool.get("function", {}).get("name")
            for tool in tools
            if tool.get("function", {}).get("name")
        }
        message = _parse_response(response_text, allowed_tools)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _request(self, prompt: str) -> str:
        with requests.Session() as session:
            session.trust_env = True
            if self.proxy_url:
                session.proxies.update(
                    {"http": self.proxy_url, "https": self.proxy_url}
                )
            with session.post(
                self.api_url,
                json={"prompt": prompt, "model": self.model},
                stream=True,
                timeout=(10, self.timeout),
            ) as response:
                if not response.ok:
                    raise RuntimeError(
                        f"Ollama API returned HTTP {response.status_code}: {response.text}"
                    )
                chunks = (
                    chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                    for chunk in response.iter_content(chunk_size=None)
                    if chunk
                )
                return "".join(chunks)

    @staticmethod
    def _render_prompt(messages: list[BaseMessage], tools: list[dict[str, Any]]) -> str:
        prompt_parts = [
            "Follow the system instructions and conversation below.",
            "When you need a tool, return exactly one JSON object in this format:",
            '{"tool": "tool_name", "arguments": {"argument": "value"}}',
            "Do not wrap the JSON in markdown. When no tool is needed, answer normally.",
        ]
        if tools:
            prompt_parts.append("Available tools:")
            for tool in tools:
                function = tool.get("function", {})
                prompt_parts.append(
                    json.dumps(
                        {
                            "name": function.get("name"),
                            "description": function.get("description", ""),
                            "parameters": function.get("parameters", {}),
                        },
                        ensure_ascii=False,
                    )
                )

        for message in messages:
            role = {
                "system": "system",
                "human": "user",
                "ai": "assistant",
                "tool": "tool result",
            }.get(message.type, message.type)
            content = message.content
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            if isinstance(message, AIMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    prompt_parts.append(
                        "assistant: "
                        + json.dumps(
                            {
                                "tool": tool_call["name"],
                                "arguments": tool_call["args"],
                            },
                            ensure_ascii=False,
                        )
                    )
            else:
                prompt_parts.append(f"{role}: {content}")
        prompt_parts.append("assistant:")
        return "\n\n".join(prompt_parts)


def _parse_response(response: str, allowed_tools: set[str]) -> AIMessage:
    parsed = _extract_json(response)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("final"), str):
            return AIMessage(content=parsed["final"])

        tool_call = parsed.get("tool_call", parsed)
        if isinstance(tool_call, dict):
            name = tool_call.get("tool", tool_call.get("name"))
            arguments = tool_call.get("arguments", tool_call.get("args", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = None
            if name in allowed_tools and isinstance(arguments, dict):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "args": arguments,
                            "id": uuid.uuid4().hex,
                            "type": "tool_call",
                        }
                    ],
                )
    return AIMessage(content=response)


def _extract_json(response: str) -> Any:
    decoder = json.JSONDecoder()
    stripped = response.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for index, character in enumerate(response):
        if character == "{":
            try:
                value, _ = decoder.raw_decode(response[index:])
                return value
            except json.JSONDecodeError:
                continue
    return None