import Link from "next/link";

import { buttonVariants } from "@/components/ui/button";
import { createAdminClient } from "@/lib/db/admin";
import { createClient } from "@/lib/db/server";

// Rendered per-request so the count reflects live DB state and so the
// build doesn't require Supabase credentials to succeed.
export const dynamic = "force-dynamic";

async function getVocabularyStats(): Promise<{ count: number | null }> {
  try {
    const supabase = createAdminClient();
    const { count, error } = await supabase
      .from("vocabulary_items")
      .select("*", { count: "exact", head: true });
    if (error) throw error;
    return { count: count ?? 0 };
  } catch {
    return { count: null };
  }
}

async function getCurrentUser() {
  try {
    const supabase = await createClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();
    return user;
  } catch {
    return null;
  }
}

export default async function Home() {
  const [{ count }, user] = await Promise.all([getVocabularyStats(), getCurrentUser()]);

  return (
    <div className="flex flex-1 flex-col items-center bg-zinc-50 px-6 py-16 dark:bg-black">
      <main className="flex w-full max-w-3xl flex-col gap-12">
        <header className="flex flex-col gap-4">
          <p className="text-xs font-medium tracking-widest text-zinc-500 uppercase dark:text-zinc-400">
            A mastery-based skill acquisition system
          </p>
          <h1 className="text-4xl font-semibold tracking-tight text-zinc-950 sm:text-5xl dark:text-zinc-50">
            Sign your way to mastery. Then leave.
          </h1>
          <p className="max-w-xl text-lg leading-relaxed text-zinc-600 dark:text-zinc-300">
            ASL vocabulary as the controlled testbed for a pedagogical architecture: spaced
            retrieval, targeted feedback, an eval-gated model, and a database state called{" "}
            <em>mastered</em> that actively removes signs from your rotation. The unit of progress
            is the sign-mastered, not the minute-spent.
          </p>
        </header>

        {!user ? (
          <section className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-3">
              <Link href="/sign-in" className={buttonVariants({ size: "lg" })}>
                Try the demo
              </Link>
              <Link href="/sign-in" className={buttonVariants({ variant: "outline", size: "lg" })}>
                Sign in with Google
              </Link>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Demo is anonymous. No email required. Your progress is saved on a throwaway session
              you can convert later.
            </p>
          </section>
        ) : (
          <section className="flex flex-wrap items-center gap-3">
            <Link href="/practice" className={buttonVariants({ size: "lg" })}>
              Practice now
            </Link>
            <Link href="/dashboard" className={buttonVariants({ variant: "outline", size: "lg" })}>
              See your dashboard
            </Link>
          </section>
        )}

        <section className="grid gap-3 sm:grid-cols-2">
          <Feature
            title="Mastery is a state, not a score"
            body="Three retrievals at intervals ≥ 7 days promote a sign to mastered. The system then removes it from active rotation. The opposite of streak mechanics."
          />
          <Feature
            title="Local inference"
            body="MediaPipe + a from-scratch BiLSTM run entirely in your browser. Your video never leaves the device."
          />
          <Feature
            title="Targeted feedback"
            body="Three hint layers: pre-attempt priming, confusion-pair-aware on fail, generic per-sign fallback. Each hint traces to a measurable signal."
          />
          <Feature
            title="Eval-gated model promotion"
            body="A model artifact does not ship unless its validation report meets every hard criterion in EVAL_GATE.md. No vibes-based AI."
          />
        </section>

        <footer className="flex flex-col gap-2 border-t border-zinc-200 pt-6 text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
          <p>
            Vocabulary live in Postgres: <strong>{count ?? "—"}</strong> signs seeded. Recognition
            path: landmark-based (ADR 0006). Sourcing: public corpora only (ADR 0004, ADR 0008).
          </p>
          <p>Built for Superbuilders. Pilot, not production. Honest about its scope.</p>
        </footer>
      </main>
    </div>
  );
}

function Feature({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex flex-col gap-1 rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
      <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">{title}</p>
      <p className="text-xs text-zinc-600 dark:text-zinc-300">{body}</p>
    </div>
  );
}
