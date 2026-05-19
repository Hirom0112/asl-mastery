"use client";

import { useState, useTransition } from "react";

import { Button } from "@/components/ui/button";
import {
  type Handedness,
  deleteAccount,
  updateFitzpatrick,
  updateHandedness,
} from "@/lib/auth/profile";

interface Props {
  handedness: string;
  fitzpatrick: number | null;
  email: string | null;
  isAnonymous: boolean;
}

const HANDEDNESS_OPTIONS: { value: Handedness; label: string }[] = [
  { value: "right", label: "Right" },
  { value: "left", label: "Left" },
  { value: "ambidextrous", label: "Ambidextrous" },
  { value: "unspecified", label: "Prefer not to say" },
];

export function SettingsForm(props: Props) {
  const [pending, startTransition] = useTransition();
  const [hand, setHand] = useState<Handedness>(props.handedness as Handedness);
  const [fp, setFp] = useState<number | null>(props.fitzpatrick);
  const [msg, setMsg] = useState<string | null>(null);

  function onHandChange(v: Handedness) {
    setHand(v);
    startTransition(async () => {
      const r = await updateHandedness(v);
      setMsg(r.error ?? "Saved");
    });
  }
  function onFpChange(v: number | null) {
    setFp(v);
    startTransition(async () => {
      const r = await updateFitzpatrick(v);
      setMsg(r.error ?? "Saved");
    });
  }

  return (
    <div className="flex flex-col gap-8">
      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-medium text-zinc-900 dark:text-zinc-100">Account</h2>
        <p className="text-sm text-zinc-600 dark:text-zinc-300">
          {props.isAnonymous
            ? "You are signed in anonymously. Your progress is saved on this account; add an email or link Google later to keep it across devices."
            : `Signed in as ${props.email ?? "(no email on file)"}.`}
        </p>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-medium text-zinc-900 dark:text-zinc-100">Handedness</h2>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          The model expects right-handed input. If you sign left-handed, we mirror the video for you
          so the model sees the convention it was trained on. Changing this only affects inference —
          your recordings are not edited.
        </p>
        <div className="flex flex-wrap gap-2">
          {HANDEDNESS_OPTIONS.map((opt) => (
            <Button
              key={opt.value}
              variant={hand === opt.value ? "default" : "outline"}
              size="sm"
              disabled={pending}
              onClick={() => onHandChange(opt.value)}
            >
              {opt.label}
            </Button>
          ))}
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
          Fitzpatrick scale (optional)
        </h2>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          We use this only for per-demographic accuracy reporting in the validation report. We never
          sell or share this. You can set it back to “prefer not to say” at any time.
        </p>
        <div className="flex flex-wrap gap-2">
          {[1, 2, 3, 4, 5, 6].map((n) => (
            <Button
              key={n}
              variant={fp === n ? "default" : "outline"}
              size="sm"
              disabled={pending}
              onClick={() => onFpChange(n)}
            >
              Type {n}
            </Button>
          ))}
          <Button
            variant={fp === null ? "default" : "outline"}
            size="sm"
            disabled={pending}
            onClick={() => onFpChange(null)}
          >
            Prefer not to say
          </Button>
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-medium text-zinc-900 dark:text-zinc-100">Delete account</h2>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          Deletes your sign-up, all attempts, and all mastery state. Cascades from{" "}
          <code>auth.users</code> through the FK chain (per docs/PRIVACY.md §5).
        </p>
        <Button
          variant="destructive"
          size="sm"
          onClick={() => startTransition(async () => void (await deleteAccount()))}
        >
          Delete my account
        </Button>
      </section>

      {msg ? (
        <p className="text-xs text-zinc-500 dark:text-zinc-400" role="status">
          {msg}
        </p>
      ) : null}
    </div>
  );
}
