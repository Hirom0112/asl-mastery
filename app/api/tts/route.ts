// Server-side OpenAI text-to-speech for the practice avatar box.
//
// Security: the OpenAI key is read from OPENAI_API_KEY (server env / .env.local)
// and NEVER reaches the client. The client sends a `phrase` KEY, not arbitrary
// text, so this cannot be abused as an open TTS proxy — only the predetermined
// phrases below are synthesizable.
//
// Voice: OpenAI "fable" at speed 1.1 (tts-1-hd). Returns audio/mpeg. If the key
// is unset (e.g. on a deploy without the env var), returns 503 and the client
// falls back to silence — the page is unaffected.

import type { NextRequest } from "next/server";

// Predetermined phrases only. Keys map to spoken copy that mirrors the avatar
// box. Keep these in sync with components/practice/runner.tsx.
const PHRASES: Record<string, string> = {
  welcome: "Welcome to your practice session. Watch the avatar, then press record and sign along.",
  pass: "Congratulations, you got it right! Let's rewatch how you did it.",
};

export async function POST(req: NextRequest) {
  const key = process.env.OPENAI_API_KEY;
  if (!key) return new Response("tts disabled (no OPENAI_API_KEY)", { status: 503 });

  let phrase = "";
  try {
    phrase = (await req.json())?.phrase ?? "";
  } catch {
    return new Response("bad request", { status: 400 });
  }
  const input = PHRASES[phrase];
  if (!input) return new Response("unknown phrase", { status: 400 });

  const upstream = await fetch("https://api.openai.com/v1/audio/speech", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "tts-1-hd",
      voice: "fable",
      input,
      speed: 1.1,
      response_format: "mp3",
    }),
  });
  if (!upstream.ok || !upstream.body) {
    return new Response("tts upstream error", { status: 502 });
  }
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "audio/mpeg",
      // phrases are static → cache the synthesized audio at the edge
      "Cache-Control": "public, max-age=86400, immutable",
    },
  });
}
