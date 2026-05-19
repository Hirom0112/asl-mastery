"use client";

import { useState, useTransition } from "react";

import { Button } from "@/components/ui/button";
import { completeOnboarding, type Handedness } from "@/lib/auth/profile";

const HANDEDNESS_OPTIONS: { value: Handedness; label: string; sub: string }[] = [
  { value: "right", label: "Right", sub: "I sign with my right hand." },
  { value: "left", label: "Left", sub: "I sign with my left hand." },
  { value: "ambidextrous", label: "Ambidextrous", sub: "Either, I'll pick later." },
  { value: "unspecified", label: "Skip for now", sub: "I'll set this in settings." },
];

export function WelcomeForm() {
  const [hand, setHand] = useState<Handedness | null>(null);
  const [pending, startTransition] = useTransition();
  const [err, setErr] = useState<string | null>(null);

  function onContinue() {
    if (!hand) {
      setErr("Pick one of the four options below — you can change it later in settings.");
      return;
    }
    startTransition(async () => {
      const r = await completeOnboarding(hand);
      if (r?.error) setErr(r.error);
    });
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-3 sm:grid-cols-2">
        {HANDEDNESS_OPTIONS.map((opt) => {
          const selected = hand === opt.value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => setHand(opt.value)}
              className={`flex flex-col items-start gap-1 rounded-xl border p-4 text-left transition ${
                selected
                  ? "border-zinc-950 bg-zinc-50 dark:border-zinc-50 dark:bg-zinc-900"
                  : "border-zinc-200 bg-white hover:border-zinc-400 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-zinc-600"
              }`}
              aria-pressed={selected}
            >
              <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                {opt.label}
              </span>
              <span className="text-xs text-zinc-500 dark:text-zinc-400">{opt.sub}</span>
            </button>
          );
        })}
      </div>

      {err ? (
        <p className="text-sm text-red-600 dark:text-red-400" role="alert">
          {err}
        </p>
      ) : null}

      <div className="flex flex-wrap items-center gap-3">
        <Button size="lg" disabled={pending} onClick={onContinue}>
          {pending ? "Starting…" : "Start practicing"}
        </Button>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          We will ask for camera access on your first attempt.
        </p>
      </div>
    </div>
  );
}
