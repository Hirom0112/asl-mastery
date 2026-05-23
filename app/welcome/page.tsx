import Link from "next/link";
import { redirect } from "next/navigation";

import { WelcomeForm } from "@/components/onboarding/welcome-form";
import { createClient } from "@/lib/db/server";
import styles from "./welcome.module.css";

export const dynamic = "force-dynamic";

const STEPS = [
  {
    n: 1,
    title: "Mastery, not a score",
    body: "Sign it right across three spaced reviews and it's mastered, then it leaves your rotation.",
  },
  {
    n: 2,
    title: "Specific feedback",
    body: "Miss one and we point to exactly what was off: handshape, location, movement, or palm.",
  },
  {
    n: 3,
    title: "Your video stays local",
    body: "Everything runs in your browser. Your camera never leaves your device.",
  },
  {
    n: 4,
    title: "Leave when you're done",
    body: "No streaks, no nudges. Master a sign and we stop asking.",
  },
];

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
    <div className={styles.root}>
      <div className={styles.wrap}>
        <nav className={styles.nav}>
          <div className={styles.brand}>
            <span className={styles.brandText}>Welcome to Mastered</span>
            <span className={styles.brandHands} aria-hidden="true" />
          </div>
          <Link href="/" className={styles.backLink}>
            ← Back home
          </Link>
        </nav>

        <main className={styles.main}>
          <div className={styles.inner}>
            <header className={styles.intro}>
              <h1 className={styles.title}>
                How this <em>works</em>.
              </h1>
              <p className={styles.subtitle}>
                Four quick things, then you&apos;re in. We get you to <em>mastered</em>, then out of
                your way.
              </p>
            </header>

            <section className={styles.steps}>
              {STEPS.map((step) => (
                <div key={step.n} className={styles.step}>
                  <p className={styles.stepNum}>Step {step.n}</p>
                  <h3 className={styles.stepTitle}>{step.title}</h3>
                  <p className={styles.stepBody}>{step.body}</p>
                </div>
              ))}
            </section>

            <section className={styles.setup}>
              <div className={styles.setupHead}>
                <h2 className={styles.setupTitle}>Quick setup</h2>
              </div>
              <WelcomeForm />
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
