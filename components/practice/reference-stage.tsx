"use client";

// Chooses the demonstration shown in the left "Sign this" stage.
//
// The native 3D-LEX signs have crisp finger-glove mocap → 3D avatar.
// The signs in VIDEO_SIGNS either re-derived their motion (SMPLest-X body + WiLoR
// fingers) and still read soft on the hands, or play the wrong body motion entirely
// (tired), or the avatar otherwise wasn't clean enough to teach from — so for THOSE
// we show a clean, real-signer Sem-Lex clip (CC BY-NC-SA) that auto-plays, loops, and
// runs at REF_PLAYBACK_RATE (slowed) so the learner can follow the handshape.
// Swap a sign out of VIDEO_SIGNS once its avatar is good.

import { SignAvatar } from "./sign-avatar";

export const VIDEO_SIGNS = new Set([
  "animal",
  "baby",
  "big",
  "bird",
  "building",
  "clean",
  "cold",
  "cool",
  "dark",
  "delicious",
  "doctor",
  "dog",
  "eat",
  "enjoy",
  "flower",
  "follow",
  "friend",
  "good",
  "help",
  "hot",
  "house",
  "kid",
  "learn",
  "money",
  "nice",
  "nothing",
  "numbers",
  "paper",
  "people",
  "pretty",
  "rain",
  "remember",
  "sick",
  "sleep",
  "sports",
  "sun",
  "swim",
  "time",
  "tired", // 3D-LEX clip plays wrong motion (hands stay at waist, never reach chest)
  "vegetable",
  "wait",
  "warm",
]);

export const REF_BASE = process.env.NEXT_PUBLIC_R2_PUBLIC_BASE_URL_REFERENCES ?? "";
// Reference clips play slowed so learners can follow the handshape (1 = real time).
export const REF_PLAYBACK_RATE = 0.6;

export function refVideoUrl(sign: string): string | null {
  const s = (sign ?? "").toLowerCase();
  return s && REF_BASE ? `${REF_BASE}/v2/${s}.webm` : null;
}

export function ReferenceStage({ signId, className }: { signId?: string; className?: string }) {
  const sign = (signId ?? "").toLowerCase();

  if (sign && REF_BASE && VIDEO_SIGNS.has(sign)) {
    return (
      <div
        className={className}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "hidden",
        }}
      >
        <video
          key={sign}
          src={`${REF_BASE}/v2/${sign}.webm`}
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
          aria-label={`reference video: ${sign}`}
          style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: 18 }}
        />
      </div>
    );
  }

  return <SignAvatar className={className} signId={sign} />;
}
