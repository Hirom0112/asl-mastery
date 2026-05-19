"use client";

import { useState, useTransition } from "react";

import { Button } from "@/components/ui/button";
import { signInAnonymously, signInWithEmail, signInWithGoogle } from "@/lib/auth/actions";

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
    <div className="flex w-full max-w-sm flex-col gap-6">
      <Button onClick={onDemo} disabled={pending} size="lg" className="w-full">
        Try the demo
      </Button>
      <p className="text-center text-xs text-zinc-500 dark:text-zinc-400">
        Anonymous account. Add an email or link Google later to keep your progress.
      </p>

      <div className="flex items-center gap-3 text-xs text-zinc-500 dark:text-zinc-400">
        <div className="h-px flex-1 bg-zinc-200 dark:bg-zinc-800" />
        or sign in
        <div className="h-px flex-1 bg-zinc-200 dark:bg-zinc-800" />
      </div>

      <Button onClick={onGoogle} disabled={pending} variant="outline" size="lg" className="w-full">
        Continue with Google
      </Button>

      {sentTo ? (
        <p className="rounded-md border border-zinc-200 bg-zinc-50 p-3 text-sm text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200">
          Magic link sent to <strong>{sentTo}</strong>. Check your inbox.
        </p>
      ) : (
        <form action={onSubmitEmail} className="flex flex-col gap-3">
          <label className="flex flex-col gap-1.5 text-sm text-zinc-700 dark:text-zinc-200">
            Email
            <input
              type="email"
              name="email"
              required
              autoComplete="email"
              placeholder="you@example.com"
              className="rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm outline-none focus:border-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:focus:border-zinc-100"
            />
          </label>
          <Button type="submit" disabled={pending} variant="secondary" size="lg" className="w-full">
            Send magic link
          </Button>
        </form>
      )}

      {message ? (
        <p className="text-sm text-red-600 dark:text-red-400" role="alert">
          {message}
        </p>
      ) : null}
    </div>
  );
}
