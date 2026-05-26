"use client";

// Tiny "watch a real signer" affordance for the coaching bubble. On any sign the
// learner can pop up the real-signer reference clip (Sem-Lex / WLASL, CC BY-NC-SA),
// looping + muted + slowed (REF_PLAYBACK_RATE) so they can follow the handshape.
// Most useful on the avatar signs, where the main demo isn't the video.

import { useEffect, useRef, useState } from "react";
import { REF_PLAYBACK_RATE, refVideoUrl } from "./reference-stage";

export function RealSignerPeek({ signId }: { signId?: string }) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLSpanElement>(null);
  const url = refVideoUrl(signId ?? "");

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // Reset when the sign changes so the popover never shows the wrong clip
  // (render-time pattern; avoids a setState-in-effect cascade).
  const [trackedSign, setTrackedSign] = useState(signId);
  if (signId !== trackedSign) {
    setTrackedSign(signId);
    setOpen(false);
  }

  if (!url) return null;

  return (
    <span ref={wrapRef} style={{ position: "relative", flexShrink: 0, display: "inline-flex" }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label="Watch a real signer demonstrate this sign"
        aria-expanded={open}
        title="Watch a real signer"
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 5,
          padding: "3px 9px",
          fontSize: 11,
          lineHeight: 1.2,
          whiteSpace: "nowrap",
          color: "#efe7d8",
          background: open ? "rgba(239,231,216,0.16)" : "transparent",
          border: "1px solid rgba(239,231,216,0.32)",
          borderRadius: 999,
          cursor: "pointer",
          fontFamily: "inherit",
        }}
      >
        <span aria-hidden="true" style={{ fontSize: 9 }}>
          ▶
        </span>
        Real signer
      </button>

      {open && (
        <span
          role="dialog"
          aria-label="Real signer demonstration"
          style={{
            position: "absolute",
            bottom: "calc(100% + 10px)",
            right: 0,
            width: 232,
            padding: 8,
            background: "#231f1a",
            border: "1px solid rgba(239,231,216,0.16)",
            borderRadius: 16,
            boxShadow: "0 22px 50px -22px rgba(0,0,0,0.7)",
            zIndex: 60,
          }}
        >
          <video
            key={url}
            src={url}
            autoPlay
            loop
            muted
            playsInline
            ref={(el) => {
              if (el) el.playbackRate = REF_PLAYBACK_RATE;
            }}
            onLoadedMetadata={(e) => {
              e.currentTarget.playbackRate = REF_PLAYBACK_RATE;
            }}
            style={{
              display: "block",
              width: "100%",
              borderRadius: 10,
              background: "#000",
              aspectRatio: "1 / 1",
              objectFit: "contain",
            }}
          />
          <span
            style={{
              display: "block",
              marginTop: 6,
              fontSize: 10.5,
              textAlign: "center",
              color: "rgba(239,231,216,0.6)",
            }}
          >
            Real signer · slowed · loops
          </span>
        </span>
      )}
    </span>
  );
}
