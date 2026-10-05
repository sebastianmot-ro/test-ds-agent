import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agents.business_understanding.agent import create_business_understanding_agent
from core.ollama_chat_model import OllamaGenerateChatModel, _parse_response
from main import run_agent_conversation


class OllamaChatModelTests(unittest.TestCase):
    def test_parses_allowed_tool_call(self) -> None:
        response = '{"tool":"ask_user","arguments":{"question":"What is the target?"}}'

        message = _parse_response(response, {"ask_user"})

        self.assertEqual(message.tool_calls[0]["name"], "ask_user")
        self.assertEqual(
            message.tool_calls[0]["args"],
            {"question": "What is the target?"},
        )

    def test_does_not_execute_unknown_tool(self) -> None:
        message = _parse_response(
            '{"tool":"delete_everything","arguments":{}}',
            {"ask_user"},
        )

        self.assertEqual(message.tool_calls, [])
        self.assertTrue(message.content)

    def test_prompt_contains_history_and_available_tools(self) -> None:
        prompt = OllamaGenerateChatModel._render_prompt(
            [HumanMessage(content="Predict sales")],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "ask_user",
                        "description": "Ask a question",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

        self.assertIn("Predict sales", prompt)
        self.assertIn("ask_user", prompt)

    def test_langgraph_executes_a_model_tool_call(self) -> None:
        llm = OllamaGenerateChatModel(
            api_url="http://localhost/generate",
            model="test-model",
            proxy_url=None,
        )
        agent = create_business_understanding_agent(llm)
        responses = [
            '{"tool":"escalate_to_orchestrator","arguments":'
            '{"blocker_summary":"Missing business goal",'
            '"suggested_next_steps":"Ask the user"}}',
            "Escalation was recorded.",
        ]

        with patch.object(OllamaGenerateChatModel, "_request", side_effect=responses):
            result = agent.invoke({"messages": [HumanMessage(content="Start project")]})

        self.assertEqual(result["messages"][-1].content, "Escalation was recorded.")

    def test_conversation_keeps_history_for_clarification(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.states = []

            def invoke(self, state):
                self.states.append(state)
                if len(self.states) == 1:
                    return {
                        "messages": [
                            HumanMessage(content="Start project"),
                            AIMessage(content="What is the target metric?"),
                        ]
                    }
                return {
                    "messages": state["messages"]
                    + [AIMessage(content="Thanks, I have the target metric.")]
                }

        agent = FakeAgent()
        with patch("builtins.input", side_effect=["Increase sales", "exit"]):
            with patch("builtins.print"):
                run_agent_conversation(agent, "Start project")

        self.assertEqual(len(agent.states), 2)
        self.assertEqual(
            [message.content for message in agent.states[1]["messages"][:2]],
            ["Start project", "What is the target metric?"],
        )
        self.assertEqual(agent.states[1]["messages"][2], ("user", "Increase sales"))


if __name__ == "__main__":
    unittest.main()