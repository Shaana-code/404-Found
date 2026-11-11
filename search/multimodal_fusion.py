"""
Module 3 | Task 3.3: Multimodal fusion logic (late fusion)

If a query has both an image and text, we run two separate vector searches
(image vs image_embedding, text vs text_embedding) and merge the two ranked
lists into one unified ranking.

Two fusion strategies are provided:
  - "weighted": weighted sum of cosine similarities (default)
  - "rrf":      Reciprocal Rank Fusion (ignores score scale, uses ranks only)

Requires: psycopg2-binary
The embedding vectors come from Module 2 (plain lists of floats).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "lost_and_found",
    "user": "postgres",
    "password": "postgres",
}


def _to_pgvector(vec: Sequence[float]) -> str:
    """Convert a list of floats to pgvector's text format: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def get_candidate_ids(
    conn,
    target_location: Optional[str] = None,
    date_from=None,
    date_to=None,
    target_category: Optional[str] = None,
) -> Optional[List[int]]:
    """Call the Task 3.1 pre-filter. Returns None if no filters were given."""
    if not any([target_location, date_from, date_to, target_category]):
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT item_id FROM prefilter_items(%s, %s, %s, %s)",
            (target_location, date_from, date_to, target_category),
        )
        return [row[0] for row in cur.fetchall()]


def _run_search(conn, fn_name: str, query_vec, candidate_ids, top_k) -> List[Tuple[int, float]]:
    """Call search_by_image / search_by_text. Returns [(item_id, similarity), ...] best-first."""
    assert fn_name in ("search_by_image", "search_by_text")
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT item_id, similarity FROM {fn_name}(%s::vector, %s::bigint[], %s)",
            (_to_pgvector(query_vec), candidate_ids, top_k),
        )
        return [(row[0], float(row[1])) for row in cur.fetchall()]


def fuse_weighted(
    image_results: List[Tuple[int, float]],
    text_results: List[Tuple[int, float]],
    image_weight: float = 0.5,
    text_weight: float = 0.5,
) -> List[Tuple[int, float]]:
    """Weighted-sum late fusion. An item missing from one list gets 0 from that modality."""
    total = image_weight + text_weight
    image_weight, text_weight = image_weight / total, text_weight / total

    scores: Dict[int, float] = {}
    for item_id, sim in image_results:
        scores[item_id] = scores.get(item_id, 0.0) + image_weight * sim
    for item_id, sim in text_results:
        scores[item_id] = scores.get(item_id, 0.0) + text_weight * sim

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fuse_rrf(
    image_results: List[Tuple[int, float]],
    text_results: List[Tuple[int, float]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across lists."""
    scores: Dict[int, float] = {}
    for results in (image_results, text_results):
        for rank, (item_id, _) in enumerate(results, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def search(
    image_vec: Optional[Sequence[float]] = None,
    text_vec: Optional[Sequence[float]] = None,
    target_location: Optional[str] = None,
    date_from=None,
    date_to=None,
    target_category: Optional[str] = None,
    top_k: int = 10,
    method: str = "weighted",
    image_weight: float = 0.5,
    text_weight: float = 0.5,
) -> List[Dict]:
    """
    Unified search entry point.
      - image only -> image search
      - text only  -> text search
      - both       -> two searches + late fusion
    Returns [{"item_id": int, "score": float}, ...] ranked best-first.
    """
    if image_vec is None and text_vec is None:
        raise ValueError("Provide at least one of image_vec or text_vec.")

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        # Step 1: pre-filter (Task 3.1)
        candidate_ids = get_candidate_ids(
            conn, target_location, date_from, date_to, target_category
        )
        if candidate_ids is not None and len(candidate_ids) == 0:
            return []

        # Fetch more than top_k from each modality so fusion has material to merge
        fetch_k = top_k * 3

        # Step 2: vector searches (Task 3.2)
        image_results = (
            _run_search(conn, "search_by_image", image_vec, candidate_ids, fetch_k)
            if image_vec is not None else []
        )
        text_results = (
            _run_search(conn, "search_by_text", text_vec, candidate_ids, fetch_k)
            if text_vec is not None else []
        )

        # Step 3: fusion
        if image_vec is not None and text_vec is not None:
            if method == "rrf":
                merged = fuse_rrf(image_results, text_results)
            else:
                merged = fuse_weighted(image_results, text_results, image_weight, text_weight)
        else:
            merged = image_results or text_results

        return [{"item_id": i, "score": s} for i, s in merged[:top_k]]
    finally:
        conn.close()


if __name__ == "__main__":
    # Demo with dummy vectors (replace with real embeddings from Module 2)
    import random

    dummy_image = [random.random() for _ in range(512)]
    dummy_text = [random.random() for _ in range(512)]

    results = search(
        image_vec=dummy_image,
        text_vec=dummy_text,
        target_location="Library",
        top_k=5,
        method="weighted",
        image_weight=0.6,
        text_weight=0.4,
    )
    for r in results:
        print(r)
