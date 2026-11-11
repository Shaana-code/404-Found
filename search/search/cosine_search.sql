-- Module 3 | Task 3.2: Cosine similarity search with pgvector
-- <=> is pgvector's cosine DISTANCE operator (0 = identical, 2 = opposite).
-- similarity = 1 - distance.
--
-- Vector size (512) must match what Module 2 produces. Change if different.

CREATE EXTENSION IF NOT EXISTS vector;

-- Approximate-nearest-neighbour indexes (HNSW) using cosine ops
CREATE INDEX IF NOT EXISTS idx_found_items_image_hnsw
    ON found_items USING hnsw (image_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_found_items_text_hnsw
    ON found_items USING hnsw (text_embedding vector_cosine_ops);

-- ---------- Image search ----------
-- candidate_ids: output of prefilter_items() as an array; NULL = no pre-filter.
CREATE OR REPLACE FUNCTION search_by_image(
    query_vec     vector(512),
    candidate_ids BIGINT[] DEFAULT NULL,
    top_k         INT      DEFAULT 10
)
RETURNS TABLE (item_id BIGINT, distance DOUBLE PRECISION, similarity DOUBLE PRECISION)
LANGUAGE sql
STABLE
AS $$
    SELECT f.item_id,
           (f.image_embedding <=> query_vec)::double precision       AS distance,
           (1 - (f.image_embedding <=> query_vec))::double precision AS similarity
    FROM found_items f
    WHERE f.image_embedding IS NOT NULL
      AND (candidate_ids IS NULL OR f.item_id = ANY(candidate_ids))
    ORDER BY f.image_embedding <=> query_vec
    LIMIT top_k;
$$;

-- ---------- Text search ----------
CREATE OR REPLACE FUNCTION search_by_text(
    query_vec     vector(512),
    candidate_ids BIGINT[] DEFAULT NULL,
    top_k         INT      DEFAULT 10
)
RETURNS TABLE (item_id BIGINT, distance DOUBLE PRECISION, similarity DOUBLE PRECISION)
LANGUAGE sql
STABLE
AS $$
    SELECT f.item_id,
           (f.text_embedding <=> query_vec)::double precision       AS distance,
           (1 - (f.text_embedding <=> query_vec))::double precision AS similarity
    FROM found_items f
    WHERE f.text_embedding IS NOT NULL
      AND (candidate_ids IS NULL OR f.item_id = ANY(candidate_ids))
    ORDER BY f.text_embedding <=> query_vec
    LIMIT top_k;
$$;

-- Example (pre-filter + search together):
-- SELECT * FROM search_by_image(
--     '[0.01, 0.02, ...]'::vector(512),
--     (SELECT array_agg(item_id) FROM prefilter_items('Library', NULL, NULL, 'bag')),
--     10
-- );
