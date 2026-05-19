import { redirect } from "next/navigation";

import { WelcomeForm } from "@/components/onboarding/welcome-form";
import { createClient } from "@/lib/db/server";

export const dynamic = "force-dynamic";

export default async function WelcomePage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/sign-in?next=/welcome");

  const { data: row } = await supabase
    .from("users")
    .select("onboarded_at")
    .eq("id", user.id)
    .maybeSingle();
  if (row?.onboarded_at) redirect("/practice");

  return (
    <div className="flex flex-1 flex-col bg-zinc-50 px-6 py-12 dark:bg-black">
      <main className="mx-auto flex w-full max-w-3xl flex-col gap-10">
        <header className="flex flex-col gap-3">
          <p className="text-xs font-medium tracking-widest text-zinc-500 uppercase dark:text-zinc-400">
            Welcome to ASL Mastery
          </p>
          <h1 className="text-3xl font-semibold tracking-tight text-zinc-950 sm:text-4xl dark:text-zinc-50">
            How this works.
          </h1>
          <p className="max-w-xl text-base leading-relaxed text-zinc-600 dark:text-zinc-300">
            Before we start, four things worth knowing. The whole point of the system is to get you
            to <em>mastered</em> on each sign — and then leave you alone.
          </p>
        </header>

        <section className="grid gap-3 sm:grid-cols-2">
          <Step
            n={1}
            title="Mastery is a state, not a score"
            body="Each sign moves through learning → in-review → mastered. Three correct reviews at growing intervals (≥ 7 days) promotes a sign to mastered and removes it from your active rotation."
          />
          <Step
            n={2}
            title="Feedback is targeted, not vibes"
            body="On a fail, we tell you what's likely wrong: the handshape, the location, the movement, the palm orientation. If the model is unsure, you get a generic per-sign hint authored from ASL-LEX phonological data."
          />
          <Step
            n={3}
            title="Your video stays local"
            body="Inference runs in your browser. Frames never leave your device. We log the attempt outcome (pass/fail, model version, time taken) to your account — not the recording."
          />
          <Step
            n={4}
            title="You leave when you're done"
            body="No streaks. No daily nudges. When you master a sign, the system celebrates and stops asking. Long-interval reviews come back only to confirm retention."
          />
        </section>

        <section className="flex flex-col gap-4 rounded-2xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
          <div className="flex flex-col gap-1">
            <h2 className="text-lg font-semibold text-zinc-950 dark:text-zinc-50">
              Quick setup: which hand do you sign with?
            </h2>
            <p className="text-sm text-zinc-600 dark:text-zinc-300">
              The model expects right-handed input. If you sign left-handed, we mirror the camera
              for you so the model sees the convention it was trained on. You can change this any
              time in settings.
            </p>
          </div>
          <WelcomeForm />
        </section>
      </main>
    </div>
  );
}

function Step({ n, title, body }: { n: number; title: string; body: string }) {
  return (
    <div className="flex flex-col gap-1 rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
      <p className="text-[10px] font-medium tracking-widest text-zinc-500 uppercase dark:text-zinc-400">
        Step {n}
      </p>
      <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">{title}</p>
      <p className="text-xs leading-relaxed text-zinc-600 dark:text-zinc-300">{body}</p>
    </div>
  );
}
