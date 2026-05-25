"use client";

// Chooses the demonstration shown in the left "Sign this" stage.
//
// The native 3D-LEX signs have crisp finger-glove mocap → 3D avatar.
// The 26 signs below either re-derived their motion (SMPLest-X body + WiLoR fingers)
// and still read soft on the hands, or (tired) play the wrong body motion entirely,
// so for THOSE we show a clean, real-signer Sem-Lex clip
// (CC BY-NC-SA) that auto-plays and loops — crisp handshapes beat soft avatar
// fingers for teaching. Swap a sign out of VIDEO_SIGNS once its avatar is good.

import { SignAvatar } from "./sign-avatar";

export const VIDEO_SIGNS = new Set([
  "animal",
  "baby",
  "big",
  "bird",
  "building",
  "cool",
  "dark",
  "eat",
  "enjoy",
  "follow",
  "friend",
  "good",
  "hot",
  "learn",
  "nothing",
  "numbers",
  "people",
  "rain",
  "sick",
  "sports",
  "sun",
  "swim",
  "time",
  "tired", // 3D-LEX clip plays wrong motion (hands stay at waist, never reach chest) → clean Sem-Lex video
  "vegetable",
  "warm",
]);

const REF_BASE = process.env.NEXT_PUBLIC_R2_PUBLIC_BASE_URL_REFERENCES ?? "";

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
          aria-label={`reference video: ${sign}`}
          style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: 18 }}
        />
      </div>
    );
  }

  return <SignAvatar className={className} signId={sign} />;
}
