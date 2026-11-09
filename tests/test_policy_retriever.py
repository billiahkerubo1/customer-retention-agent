
from src.rag.policy_retriever import retrieve_policy_chunks


def test_retriever_returns_relevant_policy():
    results = retrieve_policy_chunks(
        "customer in the first month of service needs onboarding assistance"
    )

    assert len(results) > 0
    assert results[0]["heading"] == (
        "Strategy 1: Early-Tenure Onboarding Support"
    )


def test_retriever_returns_source():
    results = retrieve_policy_chunks(
        "customer on a month-to-month contract"
    )

    assert len(results) > 0
    assert "source" in results[0]
    assert results[0]["source"] == "RET-001"


def test_retriever_rejects_empty_query():
    try:
        retrieve_policy_chunks("")
        assert False, "Expected ValueError"
    except ValueError:
        pass
