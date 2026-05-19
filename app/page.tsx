import Link from "next/link";
import type { User } from "@supabase/supabase-js";

import { buttonVariants } from "@/components/ui/button";
import { SignOutButton } from "@/components/sign-out-button";
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

async function getCurrentUser(): Promise<User | null> {
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

function userLabel(user: User): string {
  if (user.is_anonymous) return "Demo session";
  return user.email ?? user.id.slice(0, 8);
}

export default async function Home() {
  const [{ count }, user] = await Promise.all([getVocabularyStats(), getCurrentUser()]);

  return (
    <div className="flex flex-1 flex-col items-center justify-center bg-zinc-50 px-8 py-16 dark:bg-black">
      <main className="flex w-full max-w-2xl flex-col gap-10">
        <header className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium tracking-widest text-zinc-500 uppercase dark:text-zinc-400">
              ASL Mastery — pilot scaffolding
            </p>
            {user ? (
              <div className="flex items-center gap-3 text-xs text-zinc-600 dark:text-zinc-300">
                <span>{userLabel(user)}</span>
                <SignOutButton />
              </div>
            ) : (
              <Link href="/sign-in" className={buttonVariants({ variant: "outline", size: "sm" })}>
                Sign in
              </Link>
            )}
          </div>
          <h1 className="text-4xl font-semibold tracking-tight text-zinc-950 sm:text-5xl dark:text-zinc-50">
            A mastery-based skill acquisition system.
          </h1>
          <p className="max-w-xl text-lg leading-relaxed text-zinc-600 dark:text-zinc-300">
            ASL vocabulary is the controlled testbed. The unit of progress is the sign-mastered, not
            the minute-spent. Local inference, eval-gated quality, self-paced exit on mastery.
          </p>
        </header>

        <section className="flex flex-col gap-4">
          {!user ? (
            <div className="flex flex-wrap items-center gap-3">
              <Link href="/sign-in" className={buttonVariants({ size: "lg" })}>
                Try the demo
              </Link>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Anonymous session. No email required.
              </p>
            </div>
          ) : null}

          <p className="font-medium text-zinc-900 dark:text-zinc-100">
            This is Phase 5a scaffolding. The practice screen is not here yet.
          </p>
          <ul className="grid gap-1 text-sm text-zinc-600 dark:text-zinc-400">
            <li>Architecture: see docs/ARCHITECTURE.md in the repo.</li>
            <li>Recognition path: landmark-based (ADR 0006).</li>
            <li>Auth: Google + magic link + anonymous demo (ADR 0007).</li>
            <li>Pedagogical theory: docs/PEDAGOGY.md with verified citations.</li>
            <li>
              Vocabulary live in Postgres:{" "}
              <strong>{count ?? "unavailable in this environment"}</strong>{" "}
              {count !== null ? "signs seeded." : ""}
            </li>
            <li>Eval gate: docs/EVAL_GATE.md.</li>
          </ul>
        </section>

        <footer className="flex flex-col gap-2 border-t border-zinc-200 pt-6 text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
          <p>Pilot built for Superbuilders. Public-sources-only sourcing per ADR 0004.</p>
          <p>Next: Phase 3c — admin recording tool. Phase 5b — practice screen.</p>
        </footer>
      </main>
    </div>
  );
}
