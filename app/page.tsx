import Link from "next/link";

import { createAdminClient } from "@/lib/db/admin";
import { createClient } from "@/lib/db/server";
import styles from "./landing.module.css";

// Rendered per-request so the count reflects live DB state and so the
// build doesn't require Supabase credentials to succeed.
export const dynamic = "force-dynamic";

async function getVocabularyCount(): Promise<number | null> {
  try {
    const supabase = createAdminClient();
    const { count, error } = await supabase
      .from("vocabulary_items")
      .select("*", { count: "exact", head: true });
    if (error) throw error;
    return count ?? 0;
  } catch {
    return null;
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
  const [count, user] = await Promise.all([getVocabularyCount(), getCurrentUser()]);
  // Dashboard ghost button always exists; routes to the user's
  // dashboard if signed in, otherwise the sign-in entrypoint.
  const dashboardHref = user ? "/dashboard" : "/sign-in";
  // Primary CTA in nav: "Continue practicing" when signed in, "Sign up" when not.
  const navPrimaryHref = user ? "/practice" : "/sign-in";
  const navPrimaryLabel = user ? "Continue practicing" : "Sign up";
  // Hero CTA: copy stays "Continue practicing" in both states per
  // design direction; the href falls back to sign-in when signed out.
  const heroPrimaryHref = user ? "/practice" : "/sign-in";
  const vocabPhrase =
    count !== null ? `${count} vocabulary signs` : "a growing vocabulary of signs";

  return (
    <div className={styles.root}>
      <div className={styles.wrap}>
        <nav className={styles.nav}>
          <Link href="/" className={styles.logo} aria-label="ASL Mastery home">
            Mastered
            <span className={styles.logoMark} aria-hidden="true" />
          </Link>
          <div className={styles.navActions}>
            <Link href={dashboardHref} className={`${styles.btn} ${styles.btnGhost}`}>
              Dashboard
            </Link>
            <Link href={navPrimaryHref} className={`${styles.btn} ${styles.btnPrimary}`}>
              {navPrimaryLabel} <span className={styles.arrow}>→</span>
            </Link>
          </div>
        </nav>

        <section className={styles.hero}>
          <div className={styles.heroWorld} aria-hidden="true">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/world-map.webp" alt="" className={styles.heroWorldImg} />
          </div>
          <div className={styles.heroInner}>
            <div className={styles.eyebrow}>A free ASL learning platform</div>
            <h1 className={styles.heroTitle}>
              Connecting the world, <em>one sign</em> at a time.
            </h1>
            <p className={styles.heroSub}>
              Mastered is a free, browser-based home for learning American Sign Language — built so
              anyone, anywhere, can begin.
            </p>
            <div className={styles.heroCta}>
              <Link href={heroPrimaryHref} className={`${styles.btn} ${styles.btnPrimary}`}>
                Continue practicing <span className={styles.arrow}>→</span>
              </Link>
              <span className={styles.freeNote}>No card. No noise. Just sign up.</span>
            </div>
          </div>
        </section>

        <section className={styles.mission}>
          <div className={styles.missionGrid}>
            <div className={styles.beat}>
              <div className={styles.beatNum}>— 01</div>
              <h3>
                Free, for <em>everyone</em>.
              </h3>
              <p>
                No paywalls, no subscriptions, no premium tiers. Anyone with a browser and a camera
                can sign up and start learning today.
              </p>
            </div>
            <div className={styles.beat}>
              <div className={styles.beatNum}>— 02</div>
              <h3>
                So no child <em>goes without</em>.
              </h3>
              <p>
                Every child who needs ASL deserves an education in it. We exist so that access stops
                being a question of cost or geography.
              </p>
            </div>
            <div className={styles.beat}>
              <div className={styles.beatNum}>— 03</div>
              <h3>
                Mastery you <em>carry out</em>.
              </h3>
              <p>
                The app&apos;s job is to get you ready and then get out of the way. Mastered signs
                leave your rotation so your time can leave the app.
              </p>
            </div>
          </div>
        </section>

        <section className={styles.creds}>
          <div className={styles.credsRule} />
          <p>
            Built on <em>{vocabPhrase}</em> sourced from <em>WLASL</em>, the open word-level
            American Sign Language corpus, plus Lifeprint canonical references.
          </p>
          <div className={styles.credsSmall}>
            Open data. Open mission. A pilot, growing into a public good.
          </div>
        </section>

        <footer className={styles.footer}>
          <div className={styles.footerInner}>
            <p className={styles.footerTagline}>
              Connecting the world, <span>one sign</span> at a time.
            </p>
            <div className={styles.footerMeta}>
              <div className={styles.logoMini}>Mastered</div>
              <div>
                <Link href={heroPrimaryHref}>Practice</Link>
                <a href="https://github.com/Hirom0112/asl-mastery#what-is-here">Mission</a>
                <a href="https://github.com/Hirom0112/asl-mastery/blob/main/docs/PRIVACY.md">
                  Privacy
                </a>
                <span className={styles.footerStamp}>Built to solve a problem</span>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
