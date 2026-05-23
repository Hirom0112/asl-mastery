"use client";

import { useState, useTransition } from "react";

import { completeOnboarding, type Handedness } from "@/lib/auth/profile";
import styles from "@/app/welcome/welcome.module.css";

const HANDEDNESS_OPTIONS: { value: Handedness; label: string; sub: string }[] = [
  { value: "right", label: "Right", sub: "I sign with my right hand." },
  { value: "left", label: "Left", sub: "I sign with my left hand." },
];

export function WelcomeForm() {
  const [hand, setHand] = useState<Handedness | null>(null);
  const [name, setName] = useState("");
  const [pending, startTransition] = useTransition();
  const [err, setErr] = useState<string | null>(null);

  function onContinue() {
    if (!hand) {
      setErr("Pick one of the options below; you can change it later in settings.");
      return;
    }
    startTransition(async () => {
      const r = await completeOnboarding(hand, name);
      if (r?.error) setErr(r.error);
    });
  }

  return (
    <div className={styles.form}>
      <div className={styles.field}>
        <label htmlFor="displayName" className={styles.label}>
          What should we call you?
        </label>
        <input
          id="displayName"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Your name (optional)"
          maxLength={40}
          autoComplete="given-name"
          className={styles.input}
        />
      </div>

      <div className={styles.field}>
        <p className={styles.label}>Which hand do you sign with?</p>
        <div className={styles.handGrid}>
          {HANDEDNESS_OPTIONS.map((opt) => {
            const selected = hand === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => setHand(opt.value)}
                aria-pressed={selected}
                className={`${styles.handOption} ${selected ? styles.handOptionSelected : ""}`}
              >
                <span className={styles.handOptionLabel}>{opt.label}</span>
                <span className={styles.handOptionSub}>{opt.sub}</span>
              </button>
            );
          })}
        </div>
      </div>

      {err ? (
        <p className={styles.error} role="alert">
          {err}
        </p>
      ) : null}

      <div className={styles.actions}>
        <button type="button" className={styles.submit} disabled={pending} onClick={onContinue}>
          {pending ? "Starting…" : "Start practicing"}
        </button>
        <p className={styles.cameraNote}>We will ask for camera access on your first attempt.</p>
      </div>
    </div>
  );
}
