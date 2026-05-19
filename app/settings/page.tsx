import { redirect } from "next/navigation";

import { SettingsForm } from "@/components/settings/settings-form";
import { createClient } from "@/lib/db/server";

export const dynamic = "force-dynamic";

export default async function SettingsPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/sign-in?next=/settings");

  const { data: profile } = await supabase
    .from("users")
    .select("handedness, fitzpatrick, email")
    .eq("id", user.id)
    .maybeSingle();

  return (
    <div className="flex flex-1 flex-col bg-zinc-50 px-6 py-10 dark:bg-black">
      <main className="mx-auto flex w-full max-w-2xl flex-col gap-8">
        <header className="flex flex-col gap-2">
          <p className="text-xs font-medium tracking-widest text-zinc-500 uppercase dark:text-zinc-400">
            Settings
          </p>
          <h1 className="text-3xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
            Your preferences.
          </h1>
        </header>

        <SettingsForm
          handedness={profile?.handedness ?? "unspecified"}
          fitzpatrick={profile?.fitzpatrick ?? null}
          email={profile?.email ?? user.email ?? null}
          isAnonymous={!!user.is_anonymous}
        />
      </main>
    </div>
  );
}
