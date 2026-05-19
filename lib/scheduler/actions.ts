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
  flippable: boolean;
}

export async function getNextItem(): Promise<NextItem | null> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return null;

  // Load every vocabulary item + that user's mastery state in two queries.
  const [{ data: items, error: itemsErr }, { data: states, error: statesErr }] = await Promise.all([
    supabase
      .from("vocabulary_items")
      .select("id, display_gloss, category, reference_video_url, pre_attempt_hint, flippable"),
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
  mediapipeDetectionFailed: boolean;
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
      predicted_class_id: input.predictedClassId,
      confidence: input.confidence,
      passed: input.passed,
      hint_shown: input.hintShown,
      hint_source: input.hintSource,
      mediapipe_detection_failed: input.mediapipeDetectionFailed,
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

export async function flagAttempt(attemptId: string): Promise<void> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) throw new Error("not authenticated");

  // Schema gives `attempts` only select+insert grants for authenticated
  // users (no update policy). The intentional design (per
  // supabase/migrations/...rls.sql) is to route flag updates through a
  // service-role server path that re-checks ownership. Slice 1 placeholder:
  // log the flag intent; persistence requires a tiny migration in 5e to
  // grant a self-update policy on the learner_disagreed column only.
  console.warn(
    "flagAttempt: TODO wire to RLS-narrow update path; attemptId=%s user=%s",
    attemptId,
    user.id,
  );
}
