"""A dependency-light, inspectable agent with one mock knowledge-base tool."""

from __future__ import annotations

from dataclasses import dataclass

from deepeval.test_case import ToolCall

REFUND_POLICY = (
    "Unopened electronics can be returned within 30 calendar days of delivery. "
    "A receipt or order number is required. Refunds are sent to the original payment method."
)


def knowledge_base_search(query: str) -> str:
    """Return a fixed local policy so examples need no external search service."""

    _ = query
    return REFUND_POLICY


@dataclass(frozen=True, slots=True)
class AgentRun:
    input: str
    actual_output: str
    expected_output: str
    context: list[str]
    tools_called: list[ToolCall]
    expected_tools: list[ToolCall]
    steps: list[str]


def run_refund_agent(user_input: str, *, use_wrong_tool: bool = False) -> AgentRun:
    """Select a tool, use its result, and return an observable multi-step run."""

    if use_wrong_tool:
        tool_result = "A blog claims all returns are allowed for 90 days."
        tool = ToolCall(
            name="public_web_search",
            input_parameters={"query": user_input},
            output=tool_result,
        )
        actual_output = "Yes. You can return it for 90 days without proof of purchase."
    else:
        tool_result = knowledge_base_search(user_input)
        tool = ToolCall(
            name="knowledge_base_search",
            input_parameters={"query": user_input},
            output=tool_result,
        )
        actual_output = (
            "Yes—an unopened electronic item delivered 10 days ago is within the 30-day "
            "return window. Provide your receipt or order number; an approved refund goes "
            "to the original payment method."
        )

    expected_tool = ToolCall(
        name="knowledge_base_search",
        input_parameters={"query": user_input},
        output=REFUND_POLICY,
    )
    return AgentRun(
        input=user_input,
        actual_output=actual_output,
        expected_output=(
            "The unopened item is returnable within 30 days with a receipt or order number, "
            "and the refund uses the original payment method."
        ),
        context=[REFUND_POLICY],
        tools_called=[tool],
        expected_tools=[expected_tool],
        steps=[
            "Classify the request as a refund-policy question.",
            f"Call {tool.name}.",
            "Ground the final answer in the returned policy.",
        ],
    )
