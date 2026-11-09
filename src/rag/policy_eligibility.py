
def assess_retention_strategies(customer_profile):
    """
    Evaluate customer attributes against the conditions
    defined in the demonstration retention policies.

    This function does not issue offers or approve actions.
    """

    results = []

    tenure = customer_profile.get("tenure")
    contract = customer_profile.get("contract")
    customer_id = customer_profile.get("customer_id")

    # Strategy 1: Early-Tenure Onboarding Support
    if tenure is not None and 0 <= tenure <= 3:
        results.append({
            "strategy": "Early-Tenure Onboarding Support",
            "policy_id": "RET-001",
            "status": "potential_match",
            "verified_conditions": [
                f"Customer tenure is {tenure} months.",
                "Policy targets customers with 0 to 3 months of tenure."
            ],
            "unverified_conditions": [
                "Whether the customer needs onboarding assistance.",
                "Whether the customer has experienced activation "
                "or service difficulties."
            ],
            "recommended_next_step": (
                "Confirm the customer's initial service experience "
                "before recommending onboarding assistance."
            )
        })
    else:
        results.append({
            "strategy": "Early-Tenure Onboarding Support",
            "policy_id": "RET-001",
            "status": "condition_not_met",
            "verified_conditions": [],
            "unverified_conditions": [],
            "reason": (
                "The customer's tenure does not fall within "
                "the policy's 0 to 3 month range."
                if tenure is not None
                else "Customer tenure is unavailable."
            )
        })

    # Strategy 2: Contract Review and Education
    if contract == "Month-to-month":
        results.append({
            "strategy": "Contract Review and Education",
            "policy_id": "RET-001",
            "status": "potential_match",
            "verified_conditions": [
                "Customer currently has a month-to-month contract."
            ],
            "unverified_conditions": [
                "Whether the customer is interested in a longer-term "
                "contract."
            ],
            "recommended_next_step": (
                "Ask whether the customer would like to understand "
                "the available contract options."
            )
        })
    else:
        results.append({
            "strategy": "Contract Review and Education",
            "policy_id": "RET-001",
            "status": "condition_not_met",
            "verified_conditions": [],
            "unverified_conditions": [],
            "reason": (
                "Customer does not have a month-to-month contract."
                if contract is not None
                else "Customer contract information is unavailable."
            )
        })

    # Strategy 3: Technical Support Review
    results.append({
        "strategy": "Technical Support Review",
        "policy_id": "RET-001",
        "status": "requires_customer_confirmation",
        "verified_conditions": [],
        "unverified_conditions": [
            "Whether the customer has experienced technical problems.",
            "Whether the customer has unresolved support requests."
        ],
        "recommended_next_step": (
            "Ask whether the customer is experiencing technical "
            "difficulties and check support history."
        ),
        "important_note": (
            "The absence of a Tech Support subscription does not "
            "establish that the customer has a technical problem."
        )
    })

    return {
        "customer_id": customer_id,
        "policy_source": "RET-001",
        "evaluation_method": "Deterministic rule-based checks",
        "strategies": results,
        "approval_required": True
    }
