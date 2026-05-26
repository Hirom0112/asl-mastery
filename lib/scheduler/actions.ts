"use server";

// Server actions that wrap the pure scheduler in lib/scheduler.ts and
// persist transitions to Postgres via supabase-js.

import { createClient } from "@/lib/db/server";
import {
  applyFail,
  applyPass,
  DEFAULT_MASTERY,
  fromRow,
  pickNextItem,
  SCHEDULER_TUNABLES,
  toRowUpdate,
  type AttemptHistoryEntry,
  type CandidateItem,
} from "@/lib/scheduler";

export interface NextItem {
  vocabId: string;
  displayGloss: string;
  category: string;
  referenceVideoUrl: string | null;
  preAttemptHint: string | null;
  genericFailureHint: string | null;
  flippable: boolean;
}

export async function getNextItem(): Promise<NextItem | null> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return null;

  // Load every vocabulary item + that user's mastery state in two queries.
  // Order vocabulary by difficulty_rank so the scheduler's "introduce
  // next untouched sign" pick is the easiest one the learner hasn't
  // seen yet (pedagogical roadmap from L1 → L14).
  const [{ data: items, error: itemsErr }, { data: states, error: statesErr }] = await Promise.all([
    supabase
      .from("vocabulary_items")
      .select(
        "id, display_gloss, category, reference_video_url, pre_attempt_hint, generic_failure_hint, flippable, difficulty_rank",
      )
      .eq("is_active_for_practice", true)
      .order("difficulty_rank", { ascending: true, nullsFirst: false }),
    supabase.from("mastery_state").select("*").eq("user_id", user.id),
  ]);
  if (itemsErr || !items) throw itemsErr ?? new Error("no items");
  if (statesErr) throw statesErr;

  const stateByVocab = new Map((states ?? []).map((s) => [s.vocab_id, fromRow(s)]));
  const candidates: CandidateItem[] = items.map((i) => ({
    vocabId: i.id,
    state: stateByVocab.get(i.id) ?? DEFAULT_MASTERY,
  }));

  const { data: recent } = await supabase
    .from("attempts")
    .select("passed, submitted_at")
    .eq("user_id", user.id)
    .order("submitted_at", { ascending: false })
    .limit(SCHEDULER_TUNABLES.ROLLING_WINDOW);
  const history: AttemptHistoryEntry[] = (recent ?? [])
    .reverse()
    .map((a) => ({ passed: a.passed, submittedAt: new Date(a.submitted_at) }));

  const picked = pickNextItem(candidates, history, new Date());
  if (!picked) return null;

  const meta = items.find((i) => i.id === picked.vocabId);
  if (!meta) return null;
  return {
    vocabId: meta.id,
    displayGloss: meta.display_gloss,
    category: meta.category,
    referenceVideoUrl: meta.reference_video_url,
    preAttemptHint: meta.pre_attempt_hint,
    genericFailureHint: meta.generic_failure_hint,
    flippable: meta.flippable,
  };
}

// Fetch a specific vocabulary item as the next-to-practice when the
// learner overrides the scheduler from the sidebar (?sign=<id>).
// Returns null if the sign doesn't exist; the caller is expected to
// confirm the sign is *unlocked* for this user via isSignUnlocked()
// before calling this — locked signs must not be practiceable.
export async function getItemById(signId: string): Promise<NextItem | null> {
  const supabase = await createClient();
  const { data: meta, error } = await supabase
    .from("vocabulary_items")
    .select(
      "id, display_gloss, category, reference_video_url, pre_attempt_hint, generic_failure_hint, flippable",
    )
    .eq("id", signId)
    .eq("is_active_for_practice", true)
    .maybeSingle();
  if (error || !meta) return null;
  return {
    vocabId: meta.id,
    displayGloss: meta.display_gloss,
    category: meta.category,
    referenceVideoUrl: meta.reference_video_url,
    preAttemptHint: meta.pre_attempt_hint,
    genericFailureHint: meta.generic_failure_hint,
    flippable: meta.flippable,
  };
}

export interface AttemptInput {
  vocabId: string;
  promptedAtIso: string;
  submittedAtIso: string;
  predictedClassId: string | null;
  confidence: number | null;
  passed: boolean;
  hintShown: string | null;
  hintSource: "confusion_pair" | "generic_failure" | "none";
  modelVersionId: string | null;
}

export interface AttemptResult {
  attemptId: string;
  newStatus: "untouched" | "learning" | "reviewing" | "mastered";
  reachedMastery: boolean;
}

export async function recordAttempt(input: AttemptInput): Promise<AttemptResult> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) throw new Error("not authenticated");

  const now = new Date(input.submittedAtIso);

  // Read current mastery state (one row or none).
  const { data: row } = await supabase
    .from("mastery_state")
    .select("*")
    .eq("user_id", user.id)
    .eq("vocab_id", input.vocabId)
    .maybeSingle();

  const prevState = row ? fromRow(row) : DEFAULT_MASTERY;
  const nextState = input.passed ? applyPass(prevState, now) : applyFail(prevState, now);

  // Insert attempt row.
  const { data: attempt, error: attemptErr } = await supabase
    .from("attempts")
    .insert({
      user_id: user.id,
      vocab_id: input.vocabId,
      prompted_at: input.promptedAtIso,
      submitted_at: input.submittedAtIso,
      // predicted_class_id has a FK to vocabulary_items(id). When no hands are
      // detected the classifier returns "" (empty string), which is not a valid
      // id and violated the FK (Postgres 23503), throwing here and crashing the
      // page with error digest 3063673210. Coerce any empty/invalid value to
      // null (the column is nullable, FK permits null = "predicted nothing").
      predicted_class_id: input.predictedClassId || null,
      confidence: input.confidence,
      passed: input.passed,
      hint_shown: input.hintShown,
      hint_source: input.hintSource,
      model_version_id: input.modelVersionId,
    })
    .select("id")
    .single();
  if (attemptErr || !attempt) throw attemptErr ?? new Error("attempt insert failed");

  // Upsert mastery state.
  const update = toRowUpdate(nextState);
  const { error: stateErr } = await supabase.from("mastery_state").upsert({
    user_id: user.id,
    vocab_id: input.vocabId,
    ...update,
  });
  if (stateErr) throw stateErr;

  return {
    attemptId: attempt.id,
    newStatus: nextState.status,
    reachedMastery: prevState.status !== "mastered" && nextState.status === "mastered",
  };
}

export async function flagAttempt(attemptId: string): Promise<{ error?: string }> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return { error: "not authenticated" };

  // RLS policy + row-level trigger from 20260519110000_learner_disagreed.sql
  // restricts this update to the owning user and to the
  // learner_disagreed column only (no other column may transition).
  const { error } = await supabase
    .from("attempts")
    .update({ learner_disagreed: true })
    .eq("id", attemptId)
    .eq("user_id", user.id);
  if (error) return { error: error.message };
  return {};
}
