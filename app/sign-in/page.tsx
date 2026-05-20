import Link from "next/link";
import { redirect } from "next/navigation";

import { SignInForm } from "@/components/sign-in-form";
import { createClient } from "@/lib/db/server";
import styles from "./sign-in.module.css";

export const dynamic = "force-dynamic";

const ERROR_MESSAGES: Record<string, string> = {
  callback: "Sign-in could not be completed. Please try again.",
};

export default async function SignInPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  const { error } = await searchParams;

  try {
    const supabase = await createClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();
    if (user) redirect("/practice");
  } catch {
    // Env-less build context — fall through and render the form.
  }

  return (
    <div className={styles.root}>
      <div className={styles.wrap}>
        <nav className={styles.nav}>
          <Link href="/" className={styles.logo} aria-label="Mastered home">
            Mastered
            <span className={styles.logoMark} aria-hidden="true" />
          </Link>
          <Link href="/" className={styles.backLink}>
            ← Back home
          </Link>
        </nav>

        <main className={styles.main}>
          <div className={styles.card}>
            <span className={styles.eyebrow}>Sign up · Sign in</span>
            <h1 className={styles.title}>
              Start your <em>journey</em>.
            </h1>
            <p className={styles.subtitle}>
              Free, browser-based, no card. Try the demo without an account, or sign up with Google
              or email to keep your progress across sessions.
            </p>
            <SignInForm initialError={error ? ERROR_MESSAGES[error] : undefined} />
          </div>
        </main>
      </div>
    </div>
  );
}
