-- Theta Intelligence Network — Supabase schema
-- Run once: supabase db push (or paste into SQL editor)

-- ── Raw telemetry batches ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS telemetry_batches (
    id              BIGSERIAL PRIMARY KEY,
    received_at     TIMESTAMPTZ DEFAULT NOW(),
    install_id      TEXT NOT NULL,          -- anonymous sha256 of machine UUID (16 chars)
    agent_version   TEXT NOT NULL,
    batch           JSONB NOT NULL          -- array of aggregated hour-bucket records
);

CREATE INDEX IF NOT EXISTS idx_telemetry_install ON telemetry_batches(install_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_received ON telemetry_batches(received_at);

-- ── Normalized GPU health events ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS gpu_health_hourly (
    id              BIGSERIAL PRIMARY KEY,
    install_id      TEXT NOT NULL,
    hour            BIGINT NOT NULL,        -- unix epoch // 3600
    gpu_gen         TEXT NOT NULL,          -- "h100-class", "b200-class", etc.
    n_samples       INT NOT NULL,
    rtheta_mean     FLOAT,
    rtheta_std_mean FLOAT,
    ecc_sbit_total  FLOAT DEFAULT 0,
    ecc_dbit_any    BOOLEAN DEFAULT FALSE,
    clock_eff_mean  FLOAT,
    alert_types     TEXT[],
    recovery_time_p50 FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_health_gpu_gen ON gpu_health_hourly(gpu_gen);
CREATE INDEX IF NOT EXISTS idx_health_hour ON gpu_health_hourly(hour);

-- ── Community benchmark view ──────────────────────────────────────────────────
-- This is the "give back" — exposed to agents as /v1/benchmarks?gpu_gen=h100-class

CREATE OR REPLACE VIEW community_benchmarks AS
SELECT
    gpu_gen,
    COUNT(DISTINCT install_id)                          AS fleet_size,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY rtheta_mean) AS rtheta_p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY rtheta_mean) AS rtheta_p50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY rtheta_mean) AS rtheta_p75,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY rtheta_mean) AS rtheta_p95,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY clock_eff_mean) FILTER (WHERE clock_eff_mean IS NOT NULL) AS clock_eff_p50,
    AVG(ecc_sbit_total)                                AS avg_ecc_sbit_per_hour,
    SUM(CASE WHEN ecc_dbit_any THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS dbit_event_rate,
    MAX(created_at)                                    AS last_updated
FROM gpu_health_hourly
WHERE created_at > NOW() - INTERVAL '30 days'
  AND n_samples >= 5
GROUP BY gpu_gen;

-- ── Row-level security ────────────────────────────────────────────────────────
-- All writes go through the Edge Function (service role). Public can read benchmarks.

ALTER TABLE telemetry_batches ENABLE ROW LEVEL SECURITY;
ALTER TABLE gpu_health_hourly ENABLE ROW LEVEL SECURITY;

-- Edge function uses service role key — full access
-- Public (agent GET /benchmarks) reads the view only, no direct table access
