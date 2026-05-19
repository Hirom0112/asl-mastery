import { redirect } from "next/navigation";

import { SignInForm } from "@/components/sign-in-form";
import { createClient } from "@/lib/db/server";

export const dynamic = "force-dynamic";

const ERROR_MESSAGES: Record<string, string> = {
  callback: "Sign-in could not be completed. Please try again.",
};

export default async function SignInPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  const { error } = await searchParams;

  // Already signed in? Send them to the practice surface (currently
  // the home page until Phase 5b lands).
  try {
    const supabase = await createClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();
    if (user) redirect("/");
  } catch {
    // Env-less build context — fall through and render the form.
  }

  return (
    <div className="flex flex-1 flex-col items-center justify-center bg-zinc-50 px-8 py-16 dark:bg-black">
      <main className="flex w-full max-w-sm flex-col gap-8">
        <header className="flex flex-col gap-2">
          <p className="text-xs font-medium tracking-widest text-zinc-500 uppercase dark:text-zinc-400">
            ASL Mastery
          </p>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
            Sign in to practice.
          </h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-300">
            Or try the demo without an account.
          </p>
        </header>
        <SignInForm initialError={error ? ERROR_MESSAGES[error] : undefined} />
      </main>
    </div>
  );
}
