-- Module 3 | Task 3.1: Pre-filtering function (PL/pgSQL)
-- Returns candidate item IDs BEFORE vector comparison, so the vector
-- search only has to scan a small subset of rows.
--
-- ASSUMED SCHEMA (adjust names to match Module 1/2):
--   found_items(
--       item_id          BIGSERIAL PRIMARY KEY,
--       category         TEXT,
--       location         TEXT,
--       found_date       TIMESTAMPTZ,
--       status           TEXT DEFAULT 'unclaimed',
--       image_embedding  vector(512),
--       text_embedding   vector(512)
--   );

CREATE EXTENSION IF NOT EXISTS vector;

-- Indexes that make the pre-filter fast
CREATE INDEX IF NOT EXISTS idx_found_items_location ON found_items (location);
CREATE INDEX IF NOT EXISTS idx_found_items_date     ON found_items (found_date);
CREATE INDEX IF NOT EXISTS idx_found_items_category ON found_items (category);

CREATE OR REPLACE FUNCTION prefilter_items(
    target_location TEXT        DEFAULT NULL,
    date_from       TIMESTAMPTZ DEFAULT NULL,
    date_to         TIMESTAMPTZ DEFAULT NULL,
    target_category TEXT        DEFAULT NULL
)
RETURNS TABLE (item_id BIGINT)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT f.item_id
    FROM found_items f
    WHERE f.status = 'unclaimed'
      AND (target_location IS NULL OR f.location ILIKE '%' || target_location || '%')
      AND (date_from       IS NULL OR f.found_date >= date_from)
      AND (date_to         IS NULL OR f.found_date <= date_to)
      AND (target_category IS NULL OR f.category = target_category);
END;
$$;

-- Example usage:
-- SELECT * FROM prefilter_items('Library', now() - interval '7 days', now(), 'bag');
