
from src.rag.policy_retriever import retrieve_policy_chunks


TOP_K = 3

EVALUATION_CASES = [
    {
        "query": (
            "A new customer is struggling to understand "
            "how to use the service."
        ),
        "expected": [
            "Strategy 1: Early-Tenure Onboarding Support"
        ]
    },
    {
        "query": (
            "A customer on a month-to-month contract wants "
            "to understand longer-term contract options."
        ),
        "expected": [
            "Strategy 2: Contract Review and Education"
        ]
    },
    {
        "query": (
            "A customer has an unresolved internet "
            "connectivity problem."
        ),
        "expected": [
            "Strategy 3: Technical Support Review"
        ]
    },
    {
        "query": (
            "A customer requests a discount, refund, "
            "or financial compensation."
        ),
        "expected": [
            "Important Policy Boundary"
        ]
    },
    {
        "query": (
            "A customer wants to switch to a longer contract "
            "because they believe it guarantees a lower price."
        ),
        "expected": [
            "Strategy 2: Contract Review and Education"
        ]
    },
    {
        "query": (
            "A customer in their first month has activation "
            "difficulties and needs help using the service."
        ),
        "expected": [
            "Strategy 1: Early-Tenure Onboarding Support"
        ]
    },
    {
        "query": "What is the capital city of Kenya?",
        "expected": []
    },
    {
        "query": (
            "A customer who recently joined cannot activate "
            "their account and needs assistance getting started."
        ),
        "expected": [
            "Strategy 1: Early-Tenure Onboarding Support"
        ]
    },
    {
        "query": (
            "A customer's internet keeps disconnecting and "
            "their previous technical support request remains unresolved."
        ),
        "expected": [
            "Strategy 3: Technical Support Review"
        ]
    },
    {
        "query": (
            "A customer reports repeated internet outages "
            "and requests a refund for the disruption."
        ),
        "expected": [
            "Strategy 3: Technical Support Review",
            "Important Policy Boundary"
        ]
    },
    {
        "query": (
            "Can we give a customer a twenty-dollar account "
            "credit to prevent them from cancelling?"
        ),
        "expected": [
            "Important Policy Boundary"
        ]
    },
    {
        "query": (
            "A customer has a high predicted churn risk but "
            "has not reported any technical problems. Should "
            "we automatically refer them to technical support?"
        ),
        "expected": [
            "Strategy 3: Technical Support Review"
        ]
    },
    {
        "query": (
            "Does moving from a month-to-month plan to a "
            "twelve-month contract automatically guarantee "
            "a lower monthly bill?"
        ),
        "expected": [
            "Strategy 2: Contract Review and Education"
        ]
    }
]


def evaluate_retrieval():

    precision_at_k_scores = []
    returned_precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    no_match_correct = 0
    no_match_total = 0

    print("=" * 75)
    print("HYBRID RETRIEVAL + RERANKER EVALUATION")
    print(f"Total benchmark cases: {len(EVALUATION_CASES)}")
    print("=" * 75)

    for case_number, case in enumerate(
        EVALUATION_CASES,
        start=1
    ):

        query = case["query"]
        expected = set(case["expected"])

        results = retrieve_policy_chunks(
            query,
            top_k=TOP_K
        )

        retrieved = [
            result["heading"]
            for result in results
        ]

        retrieved_set = set(retrieved)

        print(f"\nCase {case_number}: {query}")
        print("-" * 75)

        if not results:
            print("No policy sections retrieved.")

        else:
            for index, result in enumerate(
                results,
                start=1
            ):

                print(
                    f"\n{index}. {result['heading']}"
                )

                print(
                    f"   RRF Score: "
                    f"{result['relevance_score']:.6f}"
                )

                print(
                    f"   BM25 Score: "
                    f"{result['bm25_score']:.4f}"
                )

                print(
                    f"   Semantic Score: "
                    f"{result['semantic_score']:.4f}"
                )

                print(
                    f"   Cross-Encoder Score: "
                    f"{result['cross_encoder_score']:.4f}"
                )

        # Evaluate queries that should have no matching policy.
        if not expected:

            no_match_total += 1

            prediction_correct = len(results) == 0

            if prediction_correct:
                no_match_correct += 1

            print("\nExpected no relevant policy: Yes")

            print(
                f"No-match prediction correct: "
                f"{prediction_correct}"
            )

            continue

        # Evaluate queries with relevant policies.
        relevant_retrieved = expected.intersection(
            retrieved_set
        )

        precision_at_k = (
            len(relevant_retrieved) / TOP_K
        )

        returned_precision = (
            len(relevant_retrieved) / len(results)
            if results else 0.0
        )

        recall = (
            len(relevant_retrieved) / len(expected)
        )

        reciprocal_rank = 0.0

        for rank, heading in enumerate(
            retrieved,
            start=1
        ):

            if heading in expected:
                reciprocal_rank = 1 / rank
                break

        precision_at_k_scores.append(
            precision_at_k
        )

        returned_precision_scores.append(
            returned_precision
        )

        recall_scores.append(recall)

        reciprocal_ranks.append(
            reciprocal_rank
        )

        print(f"\nExpected policies: {sorted(expected)}")
        print(f"Returned documents: {len(results)}")

        print(
            f"Precision@{TOP_K}: "
            f"{precision_at_k:.4f}"
        )

        print(
            f"Precision among returned: "
            f"{returned_precision:.4f}"
        )

        print(f"Recall: {recall:.4f}")

        print(
            f"Reciprocal Rank: "
            f"{reciprocal_rank:.4f}"
        )

    mean_precision_at_k = (
        sum(precision_at_k_scores)
        / len(precision_at_k_scores)
        if precision_at_k_scores else 0.0
    )

    mean_returned_precision = (
        sum(returned_precision_scores)
        / len(returned_precision_scores)
        if returned_precision_scores else 0.0
    )

    mean_recall = (
        sum(recall_scores) / len(recall_scores)
        if recall_scores else 0.0
    )

    mean_reciprocal_rank = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks else 0.0
    )

    no_match_accuracy = (
        no_match_correct / no_match_total
        if no_match_total else 0.0
    )

    print("\n" + "=" * 75)
    print("OVERALL RESULTS")
    print("=" * 75)

    print(
        f"Mean Precision@{TOP_K}: "
        f"{mean_precision_at_k:.4f}"
    )

    print(
        f"Mean Precision among returned: "
        f"{mean_returned_precision:.4f}"
    )

    print(
        f"Mean Recall: "
        f"{mean_recall:.4f}"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{mean_reciprocal_rank:.4f}"
    )

    print(
        f"No-match Accuracy: "
        f"{no_match_accuracy:.4f}"
    )

    print("=" * 75)


if __name__ == "__main__":
    evaluate_retrieval()
