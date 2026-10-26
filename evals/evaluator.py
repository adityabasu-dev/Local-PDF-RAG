
import csv
import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import load_index, retrieve, generate_answer

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "dataset.json"
RESULTS_DIR = EVAL_DIR / "results"

TOP_K = 5

ABSTENTION_PHRASES = [
    "i couldn't find",
    "i could not find",
    "cannot find this information",
    "not in the provided context",
    "not mentioned in the document",
    "not specified in the document",
    "not available in the document",
    "the document does not provide",
    "the context does not contain",
    "insufficient information"
]


def average(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def percentile(values, p):
    return float(np.percentile(values, p)) if values else None


def validate_dataset(tests):
    ids = [test["id"] for test in tests]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation IDs")

    for test in tests:
        for field in (
            "id", "question", "expected_pages",
            "expected_facts", "answerable"
        ):
            if field not in test:
                raise ValueError(
                    f"{test.get('id', '?')}: missing {field}"
                )

        if not test["question"].strip():
            raise ValueError("Empty evaluation question")

        if test["answerable"] and not test["expected_pages"]:
            raise ValueError(
                f"{test['id']}: missing ground-truth pages"
            )


def retrieval_metrics(results, expected_pages):
    if not expected_pages:
        return {
            "hit_at_k": None,
            "recall_at_k": None,
            "mrr": None,
            "ndcg_at_k": None
        }

    expected = set(expected_pages)
    retrieved = [r["page"] for r in results]

    # Page-level evaluation:
    # repeated chunks from the same page count once.
    unique_pages = list(dict.fromkeys(retrieved))

    hits = [p for p in unique_pages if p in expected]

    hit_at_k = float(bool(hits))
    recall_at_k = len(set(hits)) / len(expected)

    first_rank = next(
        (
            i for i, page in enumerate(retrieved, start=1)
            if page in expected
        ),
        None
    )

    mrr = 1 / first_rank if first_rank else 0.0

    seen = set()
    dcg = 0.0

    for rank, page in enumerate(retrieved, start=1):
        if page in expected and page not in seen:
            dcg += 1 / np.log2(rank + 1)
            seen.add(page)

    ideal_hits = min(len(expected), len(retrieved))

    idcg = sum(
        1 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    ndcg = dcg / idcg if idcg else 0.0

    return {
        "hit_at_k": hit_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "ndcg_at_k": ndcg
    }


def extract_citations(answer):
    """
    Extract simple page citations:
    (Page 90), [Page 90], Pages 90 and 97
    """
    pattern = (
        r"\bpages?\s+"
        r"(\d+(?:\s*(?:,|and|&)\s*\d+)*)"
    )

    pages = []

    for match in re.finditer(
        pattern, answer, flags=re.IGNORECASE
    ):
        pages.extend(
            int(x) for x in re.findall(r"\d+", match.group(1))
        )

    return sorted(set(pages))


def is_abstention(answer):
    normalized = " ".join(answer.lower().split())

    return any(
        phrase in normalized
        for phrase in ABSTENTION_PHRASES
    )


def keyword_coverage(answer, expected_facts):
    if not expected_facts:
        return None

    answer = answer.casefold()

    matched = sum(
        fact.casefold() in answer
        for fact in expected_facts
    )

    return matched / len(expected_facts)


def evaluate_case(test, embeddings, chunks):
    question = test["question"]

    # Retrieval
    t0 = time.perf_counter()

    retrieved = retrieve(
        question,
        embeddings,
        chunks,
        top_k=TOP_K
    )

    t1 = time.perf_counter()

    # Generation
    answer = generate_answer(
        question,
        retrieved
    )

    t2 = time.perf_counter()

    retrieved_pages = [r["page"] for r in retrieved]
    citations = extract_citations(answer)
    cited = set(citations)
    available = set(retrieved_pages)

    # This only checks citation provenance,
    # NOT whether a passage supports a claim.
    citation_provenance = (
        len(cited & available) / len(cited)
        if cited else None
    )

    rankings = retrieval_metrics(
        retrieved,
        test["expected_pages"]
    )

    abstained = is_abstention(answer)

    return {
        "id": test["id"],
        "category": test.get("category", ""),
        "question": question,
        "answerable": test["answerable"],
        "expected_pages": test["expected_pages"],
        "expected_facts": test["expected_facts"],
        "retrieved_pages": retrieved_pages,
        "retrieved_chunks": [
            {
                "id": r["id"],
                "page": r["page"],
                "score": r["score"],
                "text": r["text"]
            }
            for r in retrieved
        ],
        **rankings,
        "answer": answer,
        "citations": citations,
        "citation_provenance": citation_provenance,
        "has_citations": bool(citations),
        "abstained_detected": abstained,
        "abstention_correct": (
            not abstained if test["answerable"]
            else abstained
        ),
        "keyword_coverage": keyword_coverage(
            answer, test["expected_facts"]
        ),
        "retrieval_latency_s": round(t1 - t0, 4),
        "generation_latency_s": round(t2 - t1, 4),
        "total_latency_s": round(t2 - t0, 4)
    }


def build_summary(results):
    positives = [
        r for r in results if r["answerable"]
    ]

    negatives = [
        r for r in results if not r["answerable"]
    ]

    def mean_field(rows, key):
        return average([r[key] for r in rows])

    latencies = [r["total_latency_s"] for r in results]

    return {
        "total_tests": len(results),
        "positive_tests": len(positives),
        "negative_tests": len(negatives),
        "top_k": TOP_K,
        "retrieval": {
            "hit_at_k": mean_field(positives, "hit_at_k"),
            "recall_at_k": mean_field(
                positives, "recall_at_k"
            ),
            "mrr": mean_field(positives, "mrr"),
            "ndcg_at_k": mean_field(
                positives, "ndcg_at_k"
            )
        },
        "generation": {
            "keyword_coverage": mean_field(
                positives, "keyword_coverage"
            ),
            "answerable_non_abstention_rate": average([
                float(not r["abstained_detected"])
                for r in positives
            ]),
            "negative_abstention_rate": average([
                float(r["abstained_detected"])
                for r in negatives
            ]),
            "citation_presence_rate": average([
                float(r["has_citations"])
                for r in positives
            ]),
            "citation_provenance_rate": mean_field(
                results, "citation_provenance"
            )
        },
        "latency_seconds": {
            "retrieval_mean": mean_field(
                results, "retrieval_latency_s"
            ),
            "generation_mean": mean_field(
                results, "generation_latency_s"
            ),
            "total_mean": mean_field(
                results, "total_latency_s"
            ),
            "total_p50": percentile(latencies, 50),
            "total_p95": percentile(latencies, 95)
        }
    }


def save_reports(results, summary):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULTS_DIR / "report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "summary": summary,
                "results": results
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    with open(
        RESULTS_DIR / "summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    # Human review is required for semantic faithfulness
    with open(
        RESULTS_DIR / "manual_review.csv",
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "category", "question",
                "answer", "expected_pages",
                "retrieved_pages", "citations",
                "answer_correct_0_or_1",
                "faithful_0_or_1",
                "citations_supported_0_or_1",
                "notes"
            ]
        )

        writer.writeheader()

        for r in results:
            writer.writerow({
                "id": r["id"],
                "category": r["category"],
                "question": r["question"],
                "answer": r["answer"],
                "expected_pages": json.dumps(
                    r["expected_pages"]
                ),
                "retrieved_pages": json.dumps(
                    r["retrieved_pages"]
                ),
                "citations": json.dumps(r["citations"]),
                "answer_correct_0_or_1": "",
                "faithful_0_or_1": "",
                "citations_supported_0_or_1": "",
                "notes": ""
            })


def main():
    with open(
        DATASET_PATH, "r", encoding="utf-8"
    ) as f:
        tests = json.load(f)

    validate_dataset(tests)

    embeddings, chunks = load_index()

    if not chunks:
        raise ValueError("Saved index is empty")

    print(f"Loaded {len(chunks)} chunks")
    print(f"Running {len(tests)} evaluations")

    results = []

    for i, test in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {test['id']}")

        result = evaluate_case(test, embeddings, chunks)
        results.append(result)

        print(
            "  Retrieved:",
            result["retrieved_pages"]
        )
        print(
            "  Total latency:",
            result["total_latency_s"]
        )

    summary = build_summary(results)
    save_reports(results, summary)

    print("\n=== RAG EVALUATION RESULTS ===")
    print(json.dumps(summary, indent=2))
    print("\nReports saved to evals/results/")


if __name__ == "__main__":
    main()
