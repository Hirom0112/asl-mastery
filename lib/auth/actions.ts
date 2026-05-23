"use server";

// Auth server actions. Three entry points per ADR 0007:
// magic-link email, Google OAuth, and anonymous sign-in for the demo.
// All three settle on a Supabase session cookie that middleware.ts
// refreshes on every request.

import { headers } from "next/headers";
import { redirect } from "next/navigation";

import { createClient } from "@/lib/db/server";

export async function signInWithEmail(formData: FormData): Promise<{ error?: string }> {
  const email = String(formData.get("email") ?? "").trim();
  if (!email) return { error: "Email is required." };

  const supabase = await createClient();
  const origin = (await headers()).get("origin") ?? "";

  const { error } = await supabase.auth.signInWithOtp({
    email,
    options: { emailRedirectTo: `${origin}/auth/callback` },
  });

  if (error) return { error: error.message };
  return {};
}

export async function signInWithGoogle(): Promise<{ error?: string }> {
  const supabase = await createClient();
  const origin = (await headers()).get("origin") ?? "";

  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: "google",
    options: { redirectTo: `${origin}/auth/callback` },
  });

  if (error) return { error: error.message };
  if (data.url) redirect(data.url);
  return { error: "No OAuth redirect URL returned." };
}

export async function signInAnonymously(): Promise<{ error?: string }> {
  const supabase = await createClient();
  const { error } = await supabase.auth.signInAnonymously();
  if (error) return { error: error.message };
  // Same flow as a real sign-in: into the app. /practice sends new users
  // through /welcome (name + handedness) before the first prompt.
  redirect("/practice");
}

export async function signOut(): Promise<void> {
  const supabase = await createClient();
  await supabase.auth.signOut();
  redirect("/");
}
