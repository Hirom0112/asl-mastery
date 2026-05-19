"use server";

// Profile mutations. RLS narrows these to the authenticated user;
// service-role is not needed.

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";

import { createAdminClient } from "@/lib/db/admin";
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
  // Cascade delete chain (per docs/PRIVACY.md §5):
  //   auth.users.delete  →  public.users (FK ON DELETE CASCADE)
  //                      →  attempts     (FK ON DELETE CASCADE)
  //                      →  mastery_state (FK ON DELETE CASCADE)
  //
  // The admin call uses the service-role key (server-only). The
  // cascade is the actual deletion machinery; this server action
  // just authorizes it after verifying the caller owns the account.

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated." };

  try {
    const admin = createAdminClient();
    const { error } = await admin.auth.admin.deleteUser(user.id);
    if (error) return { error: error.message };
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) };
  }

  await supabase.auth.signOut();
  redirect("/");
}
