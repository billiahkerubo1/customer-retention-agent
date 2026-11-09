
from src.rag.policy_eligibility import assess_retention_strategies


def test_early_tenure_customer_matches_onboarding_strategy():

    customer = {
        "customer_id": "7590-VHVEG",
        "tenure": 1,
        "contract": "Month-to-month"
    }

    result = assess_retention_strategies(customer)

    onboarding = result["strategies"][0]

    assert onboarding["status"] == "potential_match"


def test_month_to_month_customer_matches_contract_strategy():

    customer = {
        "customer_id": "7590-VHVEG",
        "tenure": 1,
        "contract": "Month-to-month"
    }

    result = assess_retention_strategies(customer)

    contract = result["strategies"][1]

    assert contract["status"] == "potential_match"


def test_technical_support_requires_confirmation():

    customer = {
        "customer_id": "7590-VHVEG",
        "tenure": 1,
        "contract": "Month-to-month",
        "tech_support": "No"
    }

    result = assess_retention_strategies(customer)

    technical = result["strategies"][2]

    assert technical["status"] == "requires_customer_confirmation"


def test_long_tenure_does_not_match_onboarding_strategy():

    customer = {
        "customer_id": "TEST-001",
        "tenure": 24,
        "contract": "One year"
    }

    result = assess_retention_strategies(customer)

    onboarding = result["strategies"][0]

    assert onboarding["status"] == "condition_not_met"


def test_missing_tenure_does_not_create_onboarding_match():

    customer = {
        "customer_id": "TEST-002",
        "contract": "Month-to-month"
    }

    result = assess_retention_strategies(customer)

    onboarding = result["strategies"][0]

    assert onboarding["status"] == "condition_not_met"
