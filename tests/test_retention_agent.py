
from types import SimpleNamespace

from src.agent import retention_agent


def create_tool_call_response(function_name, function_args):
    """
    Create a simulated Gemini response containing a tool call.
    This does not contact the Gemini API.
    """

    function_call = SimpleNamespace(
        name=function_name,
        args=function_args
    )

    assistant_message = retention_agent.types.Content(
        role="model",
        parts=[
            retention_agent.types.Part.from_text(
                text="Requesting a tool."
            )
        ]
    )

    return SimpleNamespace(
        function_calls=[function_call],
        candidates=[
            SimpleNamespace(content=assistant_message)
        ],
        text=None
    )


def create_final_response(text):
    """
    Create a simulated Gemini response containing final text.
    """

    return SimpleNamespace(
        function_calls=[],
        candidates=[],
        text=text
    )


def test_agent_returns_final_response_without_tool_calls(monkeypatch):

    expected_response = "The customer has an elevated churn risk."

    monkeypatch.setattr(
        retention_agent,
        "generate_response",
        lambda conversation: create_final_response(expected_response)
    )

    result = retention_agent.run_agent(
        "Investigate customer 7590-VHVEG."
    )

    assert result == expected_response


def test_agent_executes_tool_and_returns_result_to_gemini(monkeypatch):

    tool_result = {
        "customer_id": "7590-VHVEG",
        "contract": "Month-to-month"
    }

    tool_calls = []

    def mock_customer_profile(customer_id):

        tool_calls.append(customer_id)

        return tool_result

    monkeypatch.setitem(
        retention_agent.available_functions,
        "get_customer_profile",
        mock_customer_profile
    )

    responses = [
        create_tool_call_response(
            "get_customer_profile",
            {"customer_id": "7590-VHVEG"}
        ),
        create_final_response(
            "The customer has a month-to-month contract."
        )
    ]

    received_conversations = []

    def mock_generate_response(conversation):

        received_conversations.append(conversation.copy())

        return responses.pop(0)

    monkeypatch.setattr(
        retention_agent,
        "generate_response",
        mock_generate_response
    )

    result = retention_agent.run_agent(
        "Retrieve the profile of customer 7590-VHVEG."
    )

    assert tool_calls == ["7590-VHVEG"]

    assert len(received_conversations) == 2

    assert result == (
        "The customer has a month-to-month contract."
    )

    # The second Gemini request must contain the tool result.
    second_conversation = received_conversations[1]

    assert len(second_conversation) == 3

    assert second_conversation[-1].role == "user"

    assert second_conversation[-1].parts


def test_agent_handles_unknown_tool(monkeypatch):

    responses = [
        create_tool_call_response(
            "unknown_tool",
            {}
        ),
        create_final_response(
            "I could not execute the requested tool."
        )
    ]

    received_conversations = []

    def mock_generate_response(conversation):

        received_conversations.append(conversation.copy())

        return responses.pop(0)

    monkeypatch.setattr(
        retention_agent,
        "generate_response",
        mock_generate_response
    )

    result = retention_agent.run_agent(
        "Investigate a customer."
    )

    assert result == "I could not execute the requested tool."

    assert len(received_conversations) == 2

    # The unknown tool must not crash the orchestration loop.
    assert received_conversations[1][-1].role == "user"


def test_agent_handles_tool_execution_error(monkeypatch):

    def failing_tool(customer_id):

        raise ValueError("Customer record could not be retrieved.")

    monkeypatch.setitem(
        retention_agent.available_functions,
        "get_customer_profile",
        failing_tool
    )

    responses = [
        create_tool_call_response(
            "get_customer_profile",
            {"customer_id": "7590-VHVEG"}
        ),
        create_final_response(
            "The customer profile could not be retrieved."
        )
    ]

    received_conversations = []

    def mock_generate_response(conversation):

        received_conversations.append(conversation.copy())

        return responses.pop(0)

    monkeypatch.setattr(
        retention_agent,
        "generate_response",
        mock_generate_response
    )

    result = retention_agent.run_agent(
        "Retrieve the customer profile."
    )

    assert result == (
        "The customer profile could not be retrieved."
    )

    assert len(received_conversations) == 2


def test_agent_stops_after_maximum_tool_rounds(monkeypatch):

    monkeypatch.setattr(
        retention_agent,
        "MAX_TOOL_ROUNDS",
        2
    )

    tool_calls = []

    def mock_customer_profile(customer_id):

        tool_calls.append(customer_id)

        return {
            "customer_id": customer_id
        }

    monkeypatch.setitem(
        retention_agent.available_functions,
        "get_customer_profile",
        mock_customer_profile
    )

    def mock_generate_response(conversation):

        return create_tool_call_response(
            "get_customer_profile",
            {"customer_id": "7590-VHVEG"}
        )

    monkeypatch.setattr(
        retention_agent,
        "generate_response",
        mock_generate_response
    )

    result = retention_agent.run_agent(
        "Investigate customer 7590-VHVEG."
    )

    assert len(tool_calls) == 2

    assert result == (
        "The agent reached its maximum tool-call rounds. "
        "Please try a more specific request."
    )
