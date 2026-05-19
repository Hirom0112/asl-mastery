"use server";

// Profile mutations. RLS narrows these to the authenticated user;
// service-role is not needed.

import { revalidatePath } from "next/cache";

import { createClient } from "@/lib/db/server";

export type Handedness = "right" | "left" | "ambidextrous" | "unspecified";

export async function updateHandedness(value: Handedness): Promise<{ error?: string }> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated." };

  const { error } = await supabase.from("users").update({ handedness: value }).eq("id", user.id);
  if (error) return { error: error.message };
  revalidatePath("/settings");
  revalidatePath("/practice");
  return {};
}

export async function updateFitzpatrick(value: number | null): Promise<{ error?: string }> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated." };

  if (value !== null && (value < 1 || value > 6)) {
    return { error: "Fitzpatrick value must be 1–6 or empty." };
  }

  const { error } = await supabase.from("users").update({ fitzpatrick: value }).eq("id", user.id);
  if (error) return { error: error.message };
  revalidatePath("/settings");
  return {};
}

export async function deleteAccount(): Promise<{ error?: string }> {
  // The cleanest deletion path uses Supabase Auth's deleteUser admin
  // call, which cascades into public.users via the FK ON DELETE
  // CASCADE we set up in 20260519100000_init.sql. That admin call
  // requires the service-role key and only runs server-side.
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated." };

  // TODO(phase 5e): wire to admin client + sign-out redirect.
  // Slice-1 placeholder: signing out and leaving the row in place
  // is honest behavior; a real delete needs the service-role path.
  await supabase.auth.signOut();
  return {};
}
