"use client";

// Plays predetermined OpenAI TTS phrases for the practice avatar box (via the
// server route /api/tts, which holds the key). Caches synthesized audio per
// phrase. Fails silently — no key, autoplay blocked, or network error never
// affects the practice flow.

const cache = new Map<string, string>(); // phrase -> object URL
let current: HTMLAudioElement | null = null;

export type TtsPhrase = "welcome" | "pass";

// Warm the cache (synthesize + store) without playing, so speak() is instant
// when triggered. Call on mount. Silent on any failure.
export async function prefetch(phrase: TtsPhrase): Promise<void> {
  if (typeof window === "undefined" || cache.has(phrase)) return;
  try {
    const r = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phrase }),
    });
    if (!r.ok) return;
    cache.set(phrase, URL.createObjectURL(await r.blob()));
  } catch {
    // silent
  }
}

export async function speak(phrase: TtsPhrase): Promise<void> {
  if (typeof window === "undefined") return;
  try {
    let url = cache.get(phrase);
    if (!url) {
      const r = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phrase }),
      });
      if (!r.ok) return; // 503 (no key) / 502 etc → silent
      url = URL.createObjectURL(await r.blob());
      cache.set(phrase, url);
    }
    current?.pause();
    current = new Audio(url);
    await current.play().catch(() => undefined); // autoplay policy → silent
  } catch {
    // network / other → silent
  }
}
