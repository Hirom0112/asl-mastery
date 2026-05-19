"use server";

// Resolve the currently-active model version (per ARCHITECTURE §2.4
// model_versions table). Returns null when no row has is_active=true,
// which is the slice-1 default until a model is promoted; the UI then
// falls back to stubPredict.

import { createClient } from "@/lib/db/server";

export interface ActiveModel {
  versionId: string;
  artifactUrl: string;
  configUrl: string;
  metricsUrl: string;
}

export async function getActiveModelVersion(): Promise<ActiveModel | null> {
  const supabase = await createClient();
  const { data, error } = await supabase
    .from("model_versions")
    .select("id, artifact_url, config_url, metrics_url")
    .eq("is_active", true)
    .maybeSingle();
  if (error || !data) return null;
  return {
    versionId: data.id,
    artifactUrl: data.artifact_url,
    configUrl: data.config_url,
    metricsUrl: data.metrics_url,
  };
}
