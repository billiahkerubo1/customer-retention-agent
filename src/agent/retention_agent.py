
import os
import time
import uuid
import logging
import httpx

from contextvars import ContextVar
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

from src.tools.customer_tools import (
    get_customer_profile,
    predict_customer_churn,
    explain_customer_churn,
    assess_customer_retention_eligibility,
)

from src.rag.policy_retriever import retrieve_policy_chunks


# ---------------------------------------------------------
# Logging and tracing
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

current_trace_id = ContextVar("current_trace_id", default="not-set")


def log_event(level, message, *args):
    """Log an event with the current investigation trace ID."""

    trace_id = current_trace_id.get()
    message = f"[trace_id={trace_id}] {message}"

    if level == "debug":
        logger.debug(message, *args)
    elif level == "info":
        logger.info(message, *args)
    elif level == "warning":
        logger.warning(message, *args)
    elif level == "error":
        logger.error(message, *args)
    elif level == "exception":
        logger.exception(message, *args)


# ---------------------------------------------------------
# Environment and Gemini configuration
# ---------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Check that your .env file exists and contains your API key."
    )


MODEL_NAME = "gemini-flash-latest"

# Total attempts, including the first request.
MAX_RETRIES = 2

MAX_TOOL_ROUNDS = 5
GEMINI_TIMEOUT_MS = 240_000

RETRYABLE_SERVER_ERRORS = {408, 500, 502, 503, 504}


class GeminiQuotaExceededError(Exception):
    """Raised when the Gemini API quota or rate limit is exceeded."""


class GeminiTimeoutError(Exception):
    """Raised when Gemini does not respond within the configured timeout."""


class GeminiRequestError(Exception):
    """Raised when a Gemini request fails after the allowed attempts."""

    def __init__(
        self,
        message,
        status_code=None,
        original_message=None,
    ):
        super().__init__(message)

        self.status_code = status_code
        self.original_message = original_message


client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(
        timeout=GEMINI_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(
            attempts=1,
            http_status_codes=[
                408,
                429,
                500,
                502,
                503,
                504,
            ],
        ),
    ),
)


# ---------------------------------------------------------
# Gemini function declarations
# ---------------------------------------------------------

get_customer_profile_declaration = types.FunctionDeclaration(
    name="get_customer_profile",
    description=(
        "Retrieve the available profile and account information "
        "for a customer using their customer ID."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "customer_id": types.Schema(
                type=types.Type.STRING,
                description="The customer's unique customer ID.",
            )
        },
        required=["customer_id"],
    ),
)


predict_customer_churn_declaration = types.FunctionDeclaration(
    name="predict_customer_churn",
    description=(
        "Retrieve the trained churn model's prediction for a customer. "
        "Returns the churn probability, prediction label, threshold, "
        "and model information."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "customer_id": types.Schema(
                type=types.Type.STRING,
                description="The customer's unique customer ID.",
            )
        },
        required=["customer_id"],
    ),
)


explain_customer_churn_declaration = types.FunctionDeclaration(
    name="explain_customer_churn",
    description=(
        "Explain a customer's churn prediction using SHAP. "
        "Returns the strongest feature contributions to the model output."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "customer_id": types.Schema(
                type=types.Type.STRING,
                description="The customer's unique customer ID.",
            ),
            "top_n": types.Schema(
                type=types.Type.INTEGER,
                description=(
                    "The number of top feature contributions to return. "
                    "Defaults to 5."
                ),
            ),
        },
        required=["customer_id"],
    ),
)


search_retention_policies_declaration = types.FunctionDeclaration(
    name="search_retention_policies",
    description=(
        "Search the retention policy knowledge base for relevant "
        "customer-retention strategies. Returns matching policy sections, "
        "their content, and retrieval relevance scores."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description=(
                    "A search query describing the customer situation "
                    "or retention strategy being investigated."
                ),
            ),
            "top_k": types.Schema(
                type=types.Type.INTEGER,
                description=(
                    "The maximum number of policy sections to retrieve. "
                    "Defaults to 3."
                ),
            ),
        },
        required=["query"],
    ),
)


assess_customer_retention_eligibility_declaration = (
    types.FunctionDeclaration(
        name="assess_customer_retention_eligibility",
        description=(
            "Check a customer's profile against the deterministic "
            "conditions defined in the retention policy eligibility rules. "
            "This identifies potential policy matches and conditions "
            "that require customer confirmation."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "customer_id": types.Schema(
                    type=types.Type.STRING,
                    description="The customer's unique customer ID.",
                )
            },
            required=["customer_id"],
        ),
    )
)


# ---------------------------------------------------------
# Register Gemini tools
# ---------------------------------------------------------

customer_tools = [
    types.Tool(
        function_declarations=[
            get_customer_profile_declaration,
            predict_customer_churn_declaration,
            explain_customer_churn_declaration,
            search_retention_policies_declaration,
            assess_customer_retention_eligibility_declaration,
        ]
    )
]


# ---------------------------------------------------------
# Map Gemini function names to Python functions
# ---------------------------------------------------------

available_functions = {
    "get_customer_profile": get_customer_profile,
    "predict_customer_churn": predict_customer_churn,
    "explain_customer_churn": explain_customer_churn,
    "search_retention_policies": retrieve_policy_chunks,
    "assess_customer_retention_eligibility": (
        assess_customer_retention_eligibility
    ),
}


# ---------------------------------------------------------
# Agent system instructions
# ---------------------------------------------------------

SYSTEM_INSTRUCTION = """
You are a customer retention investigation agent.

Your task is to investigate a customer's churn risk, understand the
available evidence, retrieve relevant retention policies, and recommend
appropriate next steps.

GENERAL INVESTIGATION RULES

1. Use the available tools to retrieve customer information.
2. Do not invent customer attributes, account information, predictions,
   SHAP explanations, policy content, or eligibility results.
3. Distinguish clearly between observed customer information, model
   predictions, model explanations, retrieved policy information, and
   business recommendations.
4. A churn probability is a model estimate, not certainty that a customer
   will leave.
5. The prediction threshold determines the model's classification.
   Do not treat the threshold as the churn probability itself.
6. Do not claim that a feature caused a customer to churn.
7. SHAP values explain how features contribute to the model output.
   They are not causal explanations and are not probability-point changes.
8. A positive SHAP contribution pushes the model output higher.
   A negative SHAP contribution pushes the model output lower.
9. Do not interpret a feature's meaning without considering its actual
   feature value and the feature's encoding.

POLICY AND RETRIEVAL RULES

10. Use search_retention_policies to retrieve relevant policy content.
11. A retrieval relevance score indicates similarity to the search query.
    It does not establish customer eligibility.
12. Use assess_customer_retention_eligibility to check the customer's
    profile against the deterministic eligibility conditions.
13. A potential policy match does not prove that the customer needs
    the intervention.
14. Clearly identify conditions that require customer confirmation.
15. Refer to the policy heading or source when describing a retrieved
    retention strategy.
16. Do not invent policies or claim that illustrative demo policies are
    official company policies.

COMMERCIAL AND CUSTOMER ACTION RULES

17. Never invent discounts, credits, refunds, compensation, or special
    commercial offers.
18. Do not promise a customer a price reduction or a particular outcome.
19. Follow the retrieved policy conditions and approval requirements.
20. If a recommendation requires confirmation or approval, state that
    explicitly.
21. Do not assume a customer is dissatisfied, experiencing technical
    problems, or interested in a longer contract unless the available
    evidence supports that statement.

INVESTIGATION APPROACH

22. Start by retrieving the customer's profile.
23. Retrieve the churn prediction.
24. Retrieve the SHAP explanation to understand the model's strongest
    contributing features.
25. Check the customer's retention-policy eligibility.
26. Search the knowledge base for relevant policy guidance.
27. Bring the evidence together before making a recommendation.

RESPONSE FORMAT

Present the investigation in a clear structure:

1. Customer profile
2. Churn-risk assessment
3. Model explanation
4. Relevant retention policies
5. Eligibility assessment
6. Recommended next steps
7. Important limitations or required confirmations

Keep the response grounded in tool outputs. Be clear about what is known,
what is inferred, and what still needs to be confirmed.
"""


# ---------------------------------------------------------
# Gemini error utilities
# ---------------------------------------------------------

def get_error_details(error):
    """
    Extract useful diagnostic information from a Gemini SDK exception.

    Does not log request headers, API keys, or customer information.
    """

    status_code = getattr(error, "code", None)

    error_message = (
        getattr(error, "message", None)
        or str(error)
        or type(error).__name__
    )

    error_message = str(error_message)[:500]

    return status_code, error_message


def wait_before_retry(attempt_number):
    """Apply exponential backoff before the next Gemini request."""

    wait_time = 2 ** attempt_number

    log_event(
        "warning",
        "Retrying Gemini request | next_attempt=%s/%s | wait=%s seconds",
        attempt_number + 1,
        MAX_RETRIES,
        wait_time,
    )

    time.sleep(wait_time)


# ---------------------------------------------------------
# Gemini request with retry and timeout handling
# ---------------------------------------------------------

def generate_response(conversation):
    """
    Send the current conversation to Gemini.

    Handles:
    - Retryable Gemini server errors.
    - Retryable Gemini client errors.
    - HTTP timeouts.
    - API quota and rate-limit errors.
    - Unexpected exceptions.

    The SDK is configured for one attempt so this function controls
    the application-level retry behaviour.
    """

    for attempt in range(MAX_RETRIES):

        attempt_number = attempt + 1
        request_start = time.perf_counter()

        log_event(
            "info",
            "Gemini request started | attempt=%s/%s",
            attempt_number,
            MAX_RETRIES,
        )

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=conversation,
                config=types.GenerateContentConfig(
                    tools=customer_tools,
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.2,
                ),
            )

            elapsed = time.perf_counter() - request_start

            log_event(
                "info",
                "Gemini request completed | "
                "attempt=%s/%s | duration=%.2f seconds",
                attempt_number,
                MAX_RETRIES,
                elapsed,
            )

            return response

        except errors.ServerError as error:

            elapsed = time.perf_counter() - request_start
            status_code, error_message = get_error_details(error)

            log_event(
                "error",
                "Gemini server error | "
                "status_code=%s | attempt=%s/%s | "
                "duration=%.2f seconds | error=%s",
                status_code,
                attempt_number,
                MAX_RETRIES,
                elapsed,
                error_message,
            )

            if (
                status_code not in RETRYABLE_SERVER_ERRORS
                and status_code is not None
            ):
                raise GeminiRequestError(
                    "Gemini returned a non-retryable server error. "
                    f"HTTP status: {status_code}. "
                    f"Details: {error_message}",
                    status_code=status_code,
                    original_message=error_message,
                ) from error

            if attempt == MAX_RETRIES - 1:

                raise GeminiRequestError(
                    "Gemini server request failed after the allowed retries. "
                    f"HTTP status: {status_code}. "
                    f"Details: {error_message}",
                    status_code=status_code,
                    original_message=error_message,
                ) from error

        except errors.ClientError as error:

            elapsed = time.perf_counter() - request_start
            status_code, error_message = get_error_details(error)

            if status_code == 429:

                log_event(
                    "error",
                    "Gemini quota or rate limit exceeded | "
                    "status_code=%s | duration=%.2f seconds | error=%s",
                    status_code,
                    elapsed,
                    error_message,
                )

                raise GeminiQuotaExceededError(
                    "Gemini API quota or rate limit exceeded. "
                    "Check your API usage and quota."
                ) from error

            if status_code not in RETRYABLE_SERVER_ERRORS:

                log_event(
                    "exception",
                    "Non-retryable Gemini client error | "
                    "status_code=%s | duration=%.2f seconds | error=%s",
                    status_code,
                    elapsed,
                    error_message,
                )

                raise

            log_event(
                "warning",
                "Retryable Gemini client error | "
                "status_code=%s | attempt=%s/%s | "
                "duration=%.2f seconds | error=%s",
                status_code,
                attempt_number,
                MAX_RETRIES,
                elapsed,
                error_message,
            )

            if attempt == MAX_RETRIES - 1:

                raise GeminiRequestError(
                    "Gemini request failed after the allowed retries. "
                    f"HTTP status: {status_code}. "
                    f"Details: {error_message}",
                    status_code=status_code,
                    original_message=error_message,
                ) from error

        except httpx.TimeoutException as error:

            elapsed = time.perf_counter() - request_start

            error_message = str(error) or type(error).__name__

            log_event(
                "error",
                "Gemini HTTP timeout | "
                "attempt=%s/%s | duration=%.2f seconds | "
                "error_type=%s | error=%s",
                attempt_number,
                MAX_RETRIES,
                elapsed,
                type(error).__name__,
                error_message[:500],
            )

            if attempt == MAX_RETRIES - 1:

                raise GeminiTimeoutError(
                    "Gemini did not respond within the configured "
                    "240-second timeout. Please try again later."
                ) from error

        except Exception:

            elapsed = time.perf_counter() - request_start

            log_event(
                "exception",
                "Unexpected Gemini request failure | "
                "attempt=%s/%s | duration=%.2f seconds",
                attempt_number,
                MAX_RETRIES,
                elapsed,
            )

            raise

        if attempt < MAX_RETRIES - 1:
            wait_before_retry(attempt_number)

    raise GeminiRequestError(
        "Gemini request failed after the allowed retries."
    )


# ---------------------------------------------------------
# Execute requested Python tools
# ---------------------------------------------------------

def execute_tool(function_name, function_args):
    """
    Execute a Python function requested by Gemini.

    Logs tool execution time and status without logging raw customer data.
    """

    tool_start = time.perf_counter()

    log_event(
        "info",
        "Tool execution started | tool=%s | argument_names=%s",
        function_name,
        list(function_args.keys()),
    )

    if function_name not in available_functions:

        log_event(
            "error",
            "Unknown tool requested | tool=%s",
            function_name,
        )

        return {
            "error": f"Unknown tool: {function_name}"
        }

    try:

        function_to_call = available_functions[function_name]
        result = function_to_call(**function_args)

        elapsed = time.perf_counter() - tool_start

        log_event(
            "info",
            "Tool execution completed | "
            "tool=%s | duration=%.2f seconds | status=success",
            function_name,
            elapsed,
        )

        if function_name == "search_retention_policies":

            if isinstance(result, list):

                for policy in result:

                    if isinstance(policy, dict):

                        log_event(
                            "info",
                            "Policy retrieved | heading=%s | relevance_score=%s",
                            policy.get("heading", "Unknown"),
                            policy.get("relevance_score", "Unknown"),
                        )

        return result

    except Exception as error:

        elapsed = time.perf_counter() - tool_start

        log_event(
            "exception",
            "Tool execution failed | "
            "tool=%s | duration=%.2f seconds | error_type=%s",
            function_name,
            elapsed,
            type(error).__name__,
        )

        return {
            "error": (
                f"Tool {function_name} failed: "
                f"{type(error).__name__}: {str(error)}"
            )
        }


# ---------------------------------------------------------
# Main agent orchestration
# ---------------------------------------------------------

def run_agent(user_prompt):
    """
    Run the customer-retention investigation.

    Gemini decides when to call tools and when to provide the final
    investigation response. Tool execution is bounded by MAX_TOOL_ROUNDS.
    """

    trace_id = str(uuid.uuid4())
    trace_token = current_trace_id.set(trace_id)
    investigation_start = time.perf_counter()

    log_event(
        "info",
        "Agent investigation started",
    )

    conversation = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=user_prompt)
            ],
        )
    ]

    try:

        for round_number in range(1, MAX_TOOL_ROUNDS + 1):

            print(f"\nAgent round: {round_number}")

            log_event(
                "info",
                "Agent round started | round=%s/%s",
                round_number,
                MAX_TOOL_ROUNDS,
            )

            response = generate_response(conversation)

            if response.candidates:

                model_content = response.candidates[0].content

                if model_content is not None:
                    conversation.append(model_content)

            function_calls = response.function_calls

            if not function_calls:

                final_response = response.text or (
                    "The agent completed the request but did not return "
                    "a text response."
                )

                total_elapsed = (
                    time.perf_counter() - investigation_start
                )

                log_event(
                    "info",
                    "Agent investigation completed | "
                    "rounds=%s | duration=%.2f seconds",
                    round_number,
                    total_elapsed,
                )

                return final_response

            tool_response_parts = []

            for function_call in function_calls:

                function_name = function_call.name
                function_args = dict(function_call.args or {})

                print(f"\nTool requested: {function_name}")
                print(f"Arguments: {list(function_args.keys())}")

                log_event(
                    "info",
                    "Tool requested by Gemini | "
                    "tool=%s | argument_names=%s",
                    function_name,
                    list(function_args.keys()),
                )

                result = execute_tool(
                    function_name,
                    function_args,
                )

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=function_name,
                        response={
                            "result": result
                        },
                    )
                )

            conversation.append(
                types.Content(
                    role="user",
                    parts=tool_response_parts,
                )
            )

        total_elapsed = time.perf_counter() - investigation_start

        log_event(
            "warning",
            "Maximum tool rounds reached | "
            "max_rounds=%s | duration=%.2f seconds",
            MAX_TOOL_ROUNDS,
            total_elapsed,
        )

        return (
            "The agent reached its maximum tool-call rounds. "
            "Please try a more specific request."
        )

    except Exception:

        total_elapsed = time.perf_counter() - investigation_start

        log_event(
            "exception",
            "Agent investigation failed | "
            "duration_before_failure=%.2f seconds",
            total_elapsed,
        )

        raise

    finally:

        current_trace_id.reset(trace_token)


# ---------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nCustomer Retention Investigation Agent")
    print("--------------------------------------")

    customer_id = input("\nEnter customer ID: ").strip()

    if not customer_id:

        print("\nCustomer ID cannot be empty.")

    else:

        prompt = (
            f"Investigate customer {customer_id}. "
            "Assess their churn risk, explain the model prediction, "
            "retrieve relevant retention policies, check policy "
            "eligibility, and recommend appropriate next steps. "
            "Clearly distinguish verified facts from conditions "
            "that require customer confirmation."
        )

        try:

            result = run_agent(prompt)

            print("\n")
            print("=" * 60)
            print("CUSTOMER RETENTION INVESTIGATION")
            print("=" * 60)
            print(result)
            print("=" * 60)

        except GeminiQuotaExceededError as error:

            print(f"\nGemini quota issue: {error}")

        except GeminiTimeoutError as error:

            print(f"\nGemini timeout: {error}")

        except GeminiRequestError as error:

            print(f"\nGemini request failed: {error}")

            if error.status_code is not None:
                print(f"HTTP status: {error.status_code}")

            if error.original_message:
                print(f"Original Gemini error: {error.original_message}")

        except errors.ClientError:

            print(
                "\nGemini client error. "
                "Check your API key, request configuration, "
                "and API access."
            )

            logger.exception("Gemini client error in CLI")

        except errors.ServerError:

            print(
                "\nGemini server error. "
                "The service may be temporarily unavailable."
            )

            logger.exception("Gemini server error in CLI")

        except Exception:

            print(
                "\nThe investigation encountered an unexpected error. "
                "Check the logs for details."
            )

            logger.exception("Unexpected CLI error")
