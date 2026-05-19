"use server";

// Read-only aggregations for the mastery dashboard. Keeps the
// dashboard's page server-render fast and the data shape stable.

import { createClient } from "@/lib/db/server";

export interface MasteryCounts {
  untouched: number;
  learning: number;
  reviewing: number;
  mastered: number;
  total: number;
}

export interface SignProgress {
  vocabId: string;
  displayGloss: string;
  category: string;
  status: "untouched" | "learning" | "reviewing" | "mastered";
  ease: number;
  intervalDays: number;
  nextReviewAt: string | null;
  consecutivePasses: number;
  totalAttempts: number;
  totalPasses: number;
  lastAttemptAt: string | null;
}

export async function getMasteryDashboard(): Promise<{
  counts: MasteryCounts;
  signs: SignProgress[];
}> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) {
    return {
      counts: { untouched: 0, learning: 0, reviewing: 0, mastered: 0, total: 0 },
      signs: [],
    };
  }

  const [{ data: items }, { data: states }] = await Promise.all([
    supabase.from("vocabulary_items").select("id, display_gloss, category").order("display_gloss"),
    supabase.from("mastery_state").select("*").eq("user_id", user.id),
  ]);
  const stateByVocab = new Map((states ?? []).map((s) => [s.vocab_id, s]));

  const signs: SignProgress[] = (items ?? []).map((it) => {
    const s = stateByVocab.get(it.id);
    return {
      vocabId: it.id,
      displayGloss: it.display_gloss,
      category: it.category,
      status: (s?.status as SignProgress["status"]) ?? "untouched",
      ease: s?.ease ?? 1.3,
      intervalDays: s?.interval_days ?? 0,
      nextReviewAt: s?.next_review_at ?? null,
      consecutivePasses: s?.consecutive_passes ?? 0,
      totalAttempts: s?.total_attempts ?? 0,
      totalPasses: s?.total_passes ?? 0,
      lastAttemptAt: s?.last_attempt_at ?? null,
    };
  });

  const counts: MasteryCounts = {
    untouched: 0,
    learning: 0,
    reviewing: 0,
    mastered: 0,
    total: signs.length,
  };
  for (const s of signs) counts[s.status] += 1;

  return { counts, signs };
}
