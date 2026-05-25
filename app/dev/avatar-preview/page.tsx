"use client";

// Throwaway dev route to QA avatar motion (the SMPL-X 3D + finger templates).
// Picker + prev/next over all 80 signs; deep-link with ?sign=understand.
// ?base=/templates_smplx (default), ?fingers=1 (default). Not linked anywhere.

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { SignAvatar, CAM_PRESETS } from "@/components/practice/sign-avatar";

function Inner() {
  const q = useSearchParams();
  const base = q.get("base") ?? "/templates_smplx";
  const camPos = CAM_PRESETS[q.get("cam") ?? "front"] ?? CAM_PRESETS.front;
  // QA: ?frame=N pins a deterministic pose; ?palm=data|global|off A/Bs the wrist source.
  const frameParam = q.get("frame");
  const freezeFrame = frameParam != null ? parseInt(frameParam, 10) : undefined;
  const palmMode = (q.get("palm") as "data" | "global" | "off" | null) ?? undefined;
  const mirror = (q.get("mirror") ?? "1") !== "0"; // learner-reflection; default on
  const [signs, setSigns] = useState<string[]>([]);
  const [sign, setSign] = useState(q.get("sign") ?? "understand");
  // fingers default ON now (palm orientation is constrained, so no more wrist
  // twist); handshape fidelity is approximate (curl from position-aim).
  const [fingers, setFingers] = useState((q.get("fingers") ?? "1") !== "0");

  useEffect(() => {
    fetch(`${base}/_index.json`)
      .then((r) => (r.ok ? r.json() : []))
      .then((s) => setSigns(s))
      .catch(() => {});
  }, [base]);

  const i = Math.max(0, signs.indexOf(sign));
  const go = (d: number) => {
    if (!signs.length) return;
    setSign(signs[(i + d + signs.length) % signs.length]);
  };

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#f2ede2", overflow: "hidden" }}>
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 10,
          display: "flex",
          gap: 12,
          alignItems: "center",
          padding: "10px 16px",
          fontFamily: "monospace",
          color: "#2a2620",
          background: "rgba(242,237,226,.85)",
        }}
      >
        <button onClick={() => go(-1)} style={btn}>
          ← prev
        </button>
        <button onClick={() => go(1)} style={btn}>
          next →
        </button>
        <select
          value={sign}
          onChange={(e) => setSign(e.target.value)}
          style={{ ...btn, padding: "6px 8px" }}
        >
          {signs.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <strong style={{ fontSize: 18 }}>{sign}</strong>
        <button onClick={() => setFingers((f) => !f)} style={btn}>
          fingers: {fingers ? "ON (WIP)" : "off"}
        </button>
        <span style={{ opacity: 0.6 }}>{signs.length ? `${i + 1}/${signs.length}` : "…"}</span>
      </div>
      <SignAvatar
        key={`${sign}:${fingers}:${camPos.join(",")}:${freezeFrame ?? ""}:${palmMode ?? ""}:${mirror ? 1 : 0}`}
        signId={sign}
        templatesBase={base}
        enableFingers={fingers}
        camPos={camPos}
        freezeFrame={freezeFrame}
        palmMode={palmMode}
        mirror={mirror}
        className="avatar"
      />
      <style>{`.avatar{width:100%;height:100%;}`}</style>
    </div>
  );
}

const btn: React.CSSProperties = {
  cursor: "pointer",
  border: "1px solid #8b4f2e",
  background: "#ece5d5",
  color: "#2a2620",
  borderRadius: 8,
  padding: "6px 12px",
  fontFamily: "monospace",
};

export default function AvatarPreviewPage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}
