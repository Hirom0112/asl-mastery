"use client";

import { useState, useTransition } from "react";

import { signInAnonymously, signInWithEmail, signInWithGoogle } from "@/lib/auth/actions";
import styles from "./sign-in-form.module.css";

export function SignInForm({ initialError }: { initialError?: string }) {
  const [pending, startTransition] = useTransition();
  const [message, setMessage] = useState<string | null>(initialError ?? null);
  const [sentTo, setSentTo] = useState<string | null>(null);

  function onSubmitEmail(form: FormData) {
    startTransition(async () => {
      setMessage(null);
      const { error } = await signInWithEmail(form);
      if (error) setMessage(error);
      else setSentTo(String(form.get("email") ?? ""));
    });
  }

  function onGoogle() {
    startTransition(async () => {
      setMessage(null);
      const { error } = await signInWithGoogle();
      if (error) setMessage(error);
    });
  }

  function onDemo() {
    startTransition(async () => {
      setMessage(null);
      const { error } = await signInAnonymously();
      if (error) setMessage(error);
    });
  }

  return (
    <div className={styles.root}>
      <button
        type="button"
        onClick={onDemo}
        disabled={pending}
        className={`${styles.btn} ${styles.btnPrimary}`}
      >
        Try the demo
      </button>
      <p className={styles.smallNote}>
        Anonymous account. Add an email or link Google later to keep your progress.
      </p>

      <div className={styles.divider}>or sign up</div>

      <button
        type="button"
        onClick={onGoogle}
        disabled={pending}
        className={`${styles.btn} ${styles.btnOutline}`}
      >
        Continue with Google
      </button>

      {sentTo ? (
        <p className={styles.confirmation}>
          Magic link sent to <strong>{sentTo}</strong>. Check your inbox.
        </p>
      ) : (
        <form action={onSubmitEmail} className={styles.field}>
          <label className={styles.fieldLabel} htmlFor="email">
            Email
          </label>
          <input
            id="email"
            type="email"
            name="email"
            required
            autoComplete="email"
            placeholder="you@example.com"
            className={styles.input}
          />
          <button
            type="submit"
            disabled={pending}
            className={`${styles.btn} ${styles.btnSecondary}`}
            style={{ marginTop: 8 }}
          >
            Send magic link
          </button>
        </form>
      )}

      {message ? (
        <p className={styles.errorMessage} role="alert">
          {message}
        </p>
      ) : null}
    </div>
  );
}
