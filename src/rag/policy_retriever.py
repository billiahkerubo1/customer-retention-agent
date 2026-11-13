
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "retention_strategies.md"
)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

RRF_K = 60
CANDIDATE_POOL_SIZE = 10

MIN_CROSS_ENCODER_SCORE = 0.0


# Explicit policy-intent signals.
# These are used as a safeguard when the cross-encoder
# struggles with a valid domain-specific paraphrase.

POLICY_INTENTS = {
    "Strategy 1: Early-Tenure Onboarding Support": [
        r"\bnew customer\b",
        r"\brecently joined\b",
        r"\bfirst month\b",
        r"\bearly tenure\b",
        r"\bonboarding\b",
        r"\bactivation\b",
        r"\bactivate\b",
        r"\bgetting started\b",
        r"\bhelp using\b",
        r"\bstruggling to understand\b",
        r"\bsetup assistance\b",
    ],

    "Strategy 2: Contract Review and Education": [
        r"\bmonth.to.month\b",
        r"\blonger.term\b",
        r"\blonger contract\b",
        r"\btwelve.month\b",
        r"\b12.month\b",
        r"\bcontract options\b",
        r"\blower monthly bill\b",
        r"\blower price\b",
        r"\blower monthly price\b",
        r"\bguarantee(?:d)?\b",
        r"\bcontract\b",
    ],

    "Strategy 3: Technical Support Review": [
        r"\bconnectivity\b",
        r"\binternet\b",
        r"\boutage(?:s)?\b",
        r"\bdisconnect(?:ing|s|ed)?\b",
        r"\btechnical problem(?:s)?\b",
        r"\btechnical issue(?:s)?\b",
        r"\btechnical support\b",
        r"\bunresolved support request\b",
    ],

    "Important Policy Boundary": [
        r"\bdiscount(?:s)?\b",
        r"\brefund(?:s)?\b",
        r"\bcompensation\b",
        r"\baccount credit\b",
        r"\bcredit(?:s)?\b",
        r"\btwenty dollar\b",
        r"\b20 dollar\b",
        r"\bfinancial incentive(?:s)?\b",
        r"\bfinancial compensation\b",
        r"\bwaive(?:d)? fee(?:s)?\b",
    ],
}


def load_policy_chunks():

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Policy document not found: {POLICY_PATH}"
        )

    text = POLICY_PATH.read_text(encoding="utf-8")

    sections = re.split(
        r"(?=^## )",
        text,
        flags=re.MULTILINE
    )

    chunks = []

    for section in sections:

        section = section.strip()

        if not section.startswith("## "):
            continue

        heading = (
            section.splitlines()[0]
            .replace("## ", "")
            .strip()
        )

        chunks.append({
            "source": "RET-001",
            "heading": heading,
            "content": section
        })

    if not chunks:
        raise ValueError(
            "No policy sections were found."
        )

    return chunks


def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def matches_policy_intent(query, heading):

    patterns = POLICY_INTENTS.get(heading, [])

    normalized_query = query.lower().replace("-", " ")

    return any(
        re.search(pattern, normalized_query)
        for pattern in patterns
    )


@lru_cache(maxsize=1)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


@lru_cache(maxsize=1)
def load_reranker():

    return CrossEncoder(
        RERANKER_MODEL_NAME
    )


@lru_cache(maxsize=1)
def build_retrieval_index():

    chunks = load_policy_chunks()

    documents = [
        f"{chunk['heading']}\n{chunk['content']}"
        for chunk in chunks
    ]

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    embedding_model = load_embedding_model()

    document_embeddings = embedding_model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return (
        chunks,
        documents,
        bm25,
        embedding_model,
        document_embeddings
    )


def retrieve_policy_chunks(query, top_k=3):

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    (
        chunks,
        documents,
        bm25,
        embedding_model,
        document_embeddings
    ) = build_retrieval_index()

    # 1. BM25 keyword retrieval.
    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: bm25_scores[index],
        reverse=True
    )

    bm25_ranked_indices = [
        index
        for index in bm25_ranked_indices
        if bm25_scores[index] > 0
    ]

    # 2. Semantic retrieval.
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    semantic_scores = np.dot(
        document_embeddings,
        query_embedding
    )

    semantic_ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: semantic_scores[index],
        reverse=True
    )

    # 3. Reciprocal Rank Fusion.
    fused_scores = {}

    for rank, index in enumerate(
        bm25_ranked_indices,
        start=1
    ):

        fused_scores[index] = (
            fused_scores.get(index, 0)
            + 1 / (RRF_K + rank)
        )

    for rank, index in enumerate(
        semantic_ranked_indices,
        start=1
    ):

        fused_scores[index] = (
            fused_scores.get(index, 0)
            + 1 / (RRF_K + rank)
        )

    # 4. Candidate selection.
    candidate_indices = sorted(
        fused_scores,
        key=lambda index: fused_scores[index],
        reverse=True
    )

    candidate_indices = candidate_indices[
        :CANDIDATE_POOL_SIZE
    ]

    if not candidate_indices:
        return []

    # 5. Cross-encoder reranking.
    reranker = load_reranker()

    reranking_pairs = [
        (query, documents[index])
        for index in candidate_indices
    ]

    cross_encoder_scores = reranker.predict(
        reranking_pairs,
        show_progress_bar=False
    )

    reranked_results = []

    for index, cross_score in zip(
        candidate_indices,
        cross_encoder_scores
    ):

        result = chunks[index].copy()

        result["relevance_score"] = round(
            float(fused_scores[index]),
            6
        )

        result["bm25_score"] = round(
            float(bm25_scores[index]),
            4
        )

        result["semantic_score"] = round(
            float(semantic_scores[index]),
            4
        )

        result["cross_encoder_score"] = round(
            float(cross_score),
            4
        )

        reranked_results.append(result)

    # 6. Highest cross-encoder score first.
    reranked_results.sort(
        key=lambda result: result["cross_encoder_score"],
        reverse=True
    )

    # 7. Relevance filtering.
    #
    # A result is accepted when:
    # - The cross-encoder considers it relevant, OR
    # - The query explicitly expresses an intent covered
    #   by that policy section.
    #
    # This allows valid domain-specific paraphrases while
    # preventing weak BM25/semantic overlap from admitting
    # unrelated policy sections.

    relevant_results = []

    for result in reranked_results:

        cross_encoder_relevant = (
            result["cross_encoder_score"]
            >= MIN_CROSS_ENCODER_SCORE
        )

        policy_intent_relevant = matches_policy_intent(
            query,
            result["heading"]
        )

        if cross_encoder_relevant:

            result["retrieval_method"] = (
                "cross_encoder"
            )

            relevant_results.append(result)

        elif policy_intent_relevant:

            result["retrieval_method"] = (
                "policy_intent"
            )

            relevant_results.append(result)

    return relevant_results[:top_k]


if __name__ == "__main__":

    query = (
        "A customer has been with us for one month, "
        "uses a month-to-month contract, and has "
        "an unresolved technical problem."
    )

    results = retrieve_policy_chunks(query)

    if not results:
        print("No sufficiently relevant policy found.")

    for result in results:

        print(f"\nPolicy: {result['heading']}")
        print(f"Source: {result['source']}")

        print(
            f"Retrieval Method: "
            f"{result['retrieval_method']}"
        )

        print(
            f"Cross-Encoder Score: "
            f"{result['cross_encoder_score']}"
        )

        print(
            f"BM25 Score: "
            f"{result['bm25_score']}"
        )

        print(
            f"Semantic Score: "
            f"{result['semantic_score']}"
        )

        print(result["content"])
