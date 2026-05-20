import Link from "next/link";
import { redirect } from "next/navigation";

import { PracticeRunner } from "@/components/practice/runner";
import { createClient } from "@/lib/db/server";
import { getActiveModelVersion } from "@/lib/inference/active-model";
import { getNextItem } from "@/lib/scheduler/actions";
import styles from "./practice.module.css";

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
      <div className={styles.root}>
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
              Every sign you have started is on its scheduled review interval. Come back when one is
              due — or open the dashboard to see what is on deck.
            </p>
            <Link href="/dashboard" className={`${styles.btn} ${styles.btnOutline}`}>
              View dashboard →
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const isLeftHanded = userRow.data?.handedness === "left";

  return (
    <div className={styles.root}>
      <div className={styles.wrap}>
        <PracticeRunner
          item={next}
          isLeftHanded={isLeftHanded}
          activeModelVersionId={activeModel?.versionId ?? null}
          activeModelArtifactUrl={activeModel?.artifactUrl ?? null}
          activeModelConfigUrl={activeModel?.configUrl ?? null}
        />
      </div>
    </div>
  );
}
