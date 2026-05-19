import Link from "next/link";
import { redirect } from "next/navigation";

import { PracticeRunner } from "@/components/practice/runner";
import { buttonVariants } from "@/components/ui/button";
import { createClient } from "@/lib/db/server";
import { getActiveModelVersion } from "@/lib/inference/active-model";
import { getNextItem } from "@/lib/scheduler/actions";

export const dynamic = "force-dynamic";

export default async function PracticePage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/sign-in?next=/practice");

  const [next, activeModel, userRow] = await Promise.all([
    getNextItem(),
    getActiveModelVersion(),
    supabase.from("users").select("handedness, onboarded_at").eq("id", user.id).maybeSingle(),
  ]);

  if (!userRow.data?.onboarded_at) {
    redirect("/welcome");
  }

  if (!next) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center bg-zinc-50 px-8 py-16 dark:bg-black">
        <main className="flex w-full max-w-xl flex-col gap-6 text-center">
          <h1 className="text-3xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
            Nothing to practice right now.
          </h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-300">
            Every sign you have started is on its scheduled review interval. Come back when one is
            due — or open the dashboard to see what is on deck.
          </p>
          <div className="flex justify-center">
            <Link href="/dashboard" className={buttonVariants({ size: "lg" })}>
              View dashboard
            </Link>
          </div>
        </main>
      </div>
    );
  }

  const isLeftHanded = userRow.data?.handedness === "left";

  return (
    <div className="flex flex-1 flex-col bg-zinc-50 px-6 py-10 dark:bg-black">
      <main className="mx-auto flex w-full max-w-3xl flex-col gap-6">
        <PracticeRunner
          item={next}
          isLeftHanded={isLeftHanded}
          activeModelVersionId={activeModel?.versionId ?? null}
          activeModelArtifactUrl={activeModel?.artifactUrl ?? null}
          activeModelConfigUrl={activeModel?.configUrl ?? null}
        />
      </main>
    </div>
  );
}
