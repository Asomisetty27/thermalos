// Theta Intelligence Network — Supabase Edge Function
// Deploy: supabase functions deploy telemetry_ingest
//
// Receives anonymized GPU health batches from the theta agent.
// Validates, normalizes, inserts into gpu_health_hourly.
// Returns community benchmarks for the agent's GPU generation.

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const supabase = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
);

const ALLOWED_GPU_GENS = new Set([
  "t4-class", "a100-class", "h100-class", "b200-class",
  "l40-class", "a10-class", "mi300-class", "other",
]);

function sanitize(batch: any): any | null {
  if (typeof batch !== "object" || batch === null) return null;
  const gpu_gen = String(batch.gpu_gen ?? "other");
  if (!ALLOWED_GPU_GENS.has(gpu_gen)) return null;
  const hour = Number(batch.hour);
  if (!Number.isInteger(hour) || hour < 0) return null;

  return {
    gpu_gen,
    hour,
    n_samples:         Math.min(Number(batch.n_samples ?? 0), 10000),
    rtheta_mean:       batch.rtheta_mean != null ? Number(batch.rtheta_mean) : null,
    rtheta_std_mean:   batch.rtheta_std_mean != null ? Number(batch.rtheta_std_mean) : null,
    ecc_sbit_total:    Math.max(0, Number(batch.ecc_sbit_total ?? 0)),
    ecc_dbit_any:      Boolean(batch.ecc_dbit_any),
    clock_eff_mean:    batch.clock_eff_mean != null ? Number(batch.clock_eff_mean) : null,
    alert_types:       Array.isArray(batch.alert_types) ? batch.alert_types.slice(0, 10).map(String) : [],
    recovery_time_p50: batch.recovery_time_p50 != null ? Number(batch.recovery_time_p50) : null,
  };
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  let body: any;
  try {
    body = await req.json();
  } catch {
    return new Response("Invalid JSON", { status: 400 });
  }

  const install_id    = String(body.install_id ?? "").slice(0, 32);
  const agent_version = String(body.agent_version ?? "unknown").slice(0, 20);
  const batches       = Array.isArray(body.batches) ? body.batches : [];

  if (!install_id || batches.length === 0 || batches.length > 500) {
    return new Response("Bad request", { status: 400 });
  }

  // Store raw batch for audit
  await supabase.from("telemetry_batches").insert({
    install_id,
    agent_version,
    batch: batches,
  });

  // Normalize and insert health rows
  const rows = batches
    .map((b: any) => sanitize(b))
    .filter(Boolean)
    .map((r: any) => ({ ...r, install_id }));

  if (rows.length > 0) {
    await supabase.from("gpu_health_hourly").insert(rows);
  }

  // Return community benchmarks for the GPU generations present in this batch
  const gpu_gens = [...new Set(rows.map((r: any) => r.gpu_gen))];
  const { data: benchmarks } = await supabase
    .from("community_benchmarks")
    .select("*")
    .in("gpu_gen", gpu_gens);

  return new Response(
    JSON.stringify({ accepted: rows.length, benchmarks: benchmarks ?? [] }),
    { headers: { "Content-Type": "application/json" }, status: 200 }
  );
});
