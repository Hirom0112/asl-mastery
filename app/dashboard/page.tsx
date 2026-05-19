import Link from "next/link";
import { redirect } from "next/navigation";

import { buttonVariants } from "@/components/ui/button";
import { ForgettingCurve } from "@/components/dashboard/forgetting-curve";
import { createClient } from "@/lib/db/server";
import { getMasteryDashboard, type SignProgress } from "@/lib/scheduler/queries";

export const dynamic = "force-dynamic";

function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function StatusBadge({ status }: { status: SignProgress["status"] }) {
  const styles: Record<SignProgress["status"], string> = {
    untouched: "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300",
    learning: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-200",
    reviewing: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200",
    mastered: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200",
  };
  const labels: Record<SignProgress["status"], string> = {
    untouched: "Not started",
    learning: "In learning",
    reviewing: "In review",
    mastered: "Mastered",
  };
  return (
    <span
      className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-medium ${styles[status]}`}
    >
      {labels[status]}
    </span>
  );
}

export default async function DashboardPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/sign-in?next=/dashboard");

  const { counts, signs } = await getMasteryDashboard();

  const sortedSigns = [...signs].sort((a, b) => {
    const order: Record<SignProgress["status"], number> = {
      learning: 0,
      reviewing: 1,
      untouched: 2,
      mastered: 3,
    };
    return order[a.status] - order[b.status] || a.displayGloss.localeCompare(b.displayGloss);
  });

  return (
    <div className="flex flex-1 flex-col bg-zinc-50 px-6 py-10 dark:bg-black">
      <main className="mx-auto flex w-full max-w-4xl flex-col gap-8">
        <header className="flex flex-col gap-2">
          <p className="text-xs font-medium tracking-widest text-zinc-500 uppercase dark:text-zinc-400">
            Mastery dashboard
          </p>
          <h1 className="text-3xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
            Where you stand.
          </h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-300">
            The unit of progress is the sign-mastered, not the minute-spent. Signs leave active
            rotation once mastered.
          </p>
        </header>

        <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <CountCard label="Mastered" value={counts.mastered} variant="mastered" />
          <CountCard label="In review" value={counts.reviewing} variant="reviewing" />
          <CountCard label="Learning" value={counts.learning} variant="learning" />
          <CountCard label="Not started" value={counts.untouched} variant="untouched" />
        </section>

        <div className="flex justify-start">
          <Link href="/practice" className={buttonVariants({ size: "lg" })}>
            Practice now
          </Link>
        </div>

        <section className="flex flex-col gap-3">
          <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-200">
            Per-sign progress
          </h2>
          <div className="grid grid-cols-1 gap-2">
            {sortedSigns.map((sign) => (
              <div
                key={sign.vocabId}
                className="flex items-center justify-between rounded-lg border border-zinc-200 bg-white px-4 py-3 text-sm dark:border-zinc-800 dark:bg-zinc-900"
              >
                <div className="flex flex-col">
                  <span className="font-medium text-zinc-900 dark:text-zinc-50">
                    {sign.displayGloss}
                  </span>
                  <span className="text-xs text-zinc-500 dark:text-zinc-400">{sign.category}</span>
                </div>
                <div className="flex items-center gap-4">
                  <ForgettingCurve sign={sign} />
                  <div className="flex flex-col items-end gap-1">
                    <StatusBadge status={sign.status} />
                    <span className="text-[10px] text-zinc-500 dark:text-zinc-400">
                      next review {fmtDate(sign.nextReviewAt)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

function CountCard({
  label,
  value,
  variant,
}: {
  label: string;
  value: number;
  variant: SignProgress["status"];
}) {
  const styles: Record<SignProgress["status"], string> = {
    mastered: "border-emerald-200 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950",
    reviewing: "border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-950",
    learning: "border-blue-200 bg-blue-50 dark:border-blue-900 dark:bg-blue-950",
    untouched: "border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900",
  };
  return (
    <div className={`flex flex-col gap-1 rounded-xl border p-4 ${styles[variant]}`}>
      <span className="text-xs text-zinc-600 dark:text-zinc-300">{label}</span>
      <span className="text-2xl font-semibold text-zinc-950 dark:text-zinc-50">{value}</span>
    </div>
  );
}
