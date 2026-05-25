import Link from "next/link";
import { redirect } from "next/navigation";

import { PracticeRunner } from "@/components/practice/runner";
import { PracticeSidebar } from "@/components/practice/sidebar";
import { createClient } from "@/lib/db/server";
import { getActiveModelVersion } from "@/lib/inference/active-model";
import { getItemById, getNextItem } from "@/lib/scheduler/actions";
import { getVocabProgression, isSignUnlocked } from "@/lib/scheduler/vocab-progression";
import styles from "./practice.module.css";

export const dynamic = "force-dynamic";

interface PracticePageProps {
  searchParams: Promise<{ sign?: string; skip?: string }>;
}

export default async function PracticePage({ searchParams }: PracticePageProps) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/sign-in?next=/practice");

  const params = await searchParams;
  const requestedSign = params.sign;
  // ?skip=1 means the learner pressed "Skip for now" (after ≥3 misses) — let
  // them advance to the next sign even though it's still locked by progression.
  const isSkip = params.skip === "1";

  // Resolve the sign to practice:
  //   ?sign=<id> → sidebar override (unlocked-only) or an explicit skip (any).
  //   otherwise → scheduler picks the next due item.
  let next;
  if (requestedSign) {
    const allowed = isSkip || (await isSignUnlocked(requestedSign));
    if (allowed) {
      next = await getItemById(requestedSign);
    }
  }
  if (!next) {
    next = await getNextItem();
  }

  const [activeModel, userRow, progression] = await Promise.all([
    getActiveModelVersion(),
    supabase.from("users").select("handedness, onboarded_at").eq("id", user.id).maybeSingle(),
    getVocabProgression(),
  ]);

  if (!userRow.data?.onboarded_at) {
    redirect("/welcome");
  }

  if (!next) {
    return (
      <div className={styles.root}>
        <div className={styles.layout}>
          <PracticeSidebar items={progression} currentSignId="" />
          <div className={styles.wrap}>
            <div className={styles.empty}>
              <p className={styles.eyebrow}>All clear</p>
              <h1 className={styles.gloss}>
                Nothing to practice <em>right now</em>.
              </h1>
              <p
                style={{
                  fontFamily: "var(--serif)",
                  fontStyle: "italic",
                  color: "var(--ink-soft)",
                  marginBottom: 28,
                }}
              >
                Every sign you have started is on its scheduled review interval. Pick one from the
                sidebar, or come back when one is due.
              </p>
              <Link href="/dashboard" className={`${styles.btn} ${styles.btnOutline}`}>
                View dashboard →
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const isLeftHanded = userRow.data?.handedness === "left";

  // The next sign by difficulty rank (progression is rank-ordered) — the target
  // for "Skip for now". Null if the current sign is the last one.
  const curIdx = progression.findIndex((p) => p.id === next.vocabId);
  const nextSignId = curIdx >= 0 ? (progression[curIdx + 1]?.id ?? null) : null;

  return (
    <div className={styles.root}>
      <div className={styles.layout}>
        <PracticeSidebar
          items={progression}
          currentSignId={next.vocabId}
          greetingName={
            typeof user.user_metadata?.display_name === "string"
              ? user.user_metadata.display_name
              : undefined
          }
        />
        <div className={styles.wrap}>
          <PracticeRunner
            item={next}
            isLeftHanded={isLeftHanded}
            nextSignId={nextSignId}
            activeModelVersionId={activeModel?.versionId ?? null}
            activeModelArtifactUrl={activeModel?.artifactUrl ?? null}
            activeModelConfigUrl={activeModel?.configUrl ?? null}
          />
        </div>
      </div>
    </div>
  );
}
