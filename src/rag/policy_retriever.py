
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "retention_strategies.md"
)


def load_policy_chunks():
    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Policy document not found: {POLICY_PATH}"
        )

    text = POLICY_PATH.read_text(encoding="utf-8")

    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    chunks = []

    for section in sections:
        section = section.strip()

        if not section.startswith("## "):
            continue

        heading = section.splitlines()[0].replace("## ", "").strip()

        chunks.append({
            "source": "RET-001",
            "heading": heading,
            "content": section
        })

    if not chunks:
        raise ValueError("No policy sections were found.")

    return chunks


def retrieve_policy_chunks(query, top_k=3):
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    chunks = load_policy_chunks()

    documents = [chunk["content"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    document_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        score = float(similarities[index])

        if score <= 0:
            continue

        result = chunks[index].copy()
        result["relevance_score"] = round(score, 4)

        results.append(result)

    return results


if __name__ == "__main__":
    query = (
        "A customer has been with us for one month, "
        "uses a month-to-month contract, and has no technical support."
    )

    results = retrieve_policy_chunks(query)

    for result in results:
        print(f"\nPolicy: {result['heading']}")
        print(f"Source: {result['source']}")
        print(f"Relevance: {result['relevance_score']}")
        print(result["content"])
