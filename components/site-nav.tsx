import Link from "next/link";
import { headers } from "next/headers";

import { buttonVariants } from "@/components/ui/button";
import { SignOutButton } from "@/components/sign-out-button";
import { createClient } from "@/lib/db/server";

// Routes that ship their own nav and should NOT also render SiteNav.
const SUPPRESS_ON: ReadonlySet<string> = new Set(["/"]);

export async function SiteNav() {
  const pathname = (await headers()).get("x-pathname") ?? "";
  if (SUPPRESS_ON.has(pathname)) return null;

  let user: { is_anonymous?: boolean; email?: string | null; id?: string } | null = null;
  try {
    const supabase = await createClient();
    const res = await supabase.auth.getUser();
    user = res.data.user;
  } catch {
    user = null;
  }

  return (
    <header className="border-b border-zinc-200 bg-white/80 backdrop-blur dark:border-zinc-800 dark:bg-black/60">
      <nav
        className="mx-auto flex w-full max-w-5xl items-center justify-between gap-4 px-6 py-3 text-sm"
        aria-label="Primary"
      >
        <Link
          href="/"
          className="font-semibold tracking-tight text-zinc-950 dark:text-zinc-50"
          aria-label="ASL Mastery home"
        >
          ASL Mastery
        </Link>
        {user ? (
          <div className="flex items-center gap-3 text-xs text-zinc-600 dark:text-zinc-300">
            <Link href="/practice" className="hover:text-zinc-950 dark:hover:text-zinc-50">
              Practice
            </Link>
            <Link href="/dashboard" className="hover:text-zinc-950 dark:hover:text-zinc-50">
              Dashboard
            </Link>
            <Link href="/settings" className="hover:text-zinc-950 dark:hover:text-zinc-50">
              Settings
            </Link>
            <span aria-hidden="true" className="text-zinc-300 dark:text-zinc-700">
              ·
            </span>
            <span>
              {user.is_anonymous
                ? "Demo session"
                : (user.email ?? user.id?.slice(0, 8) ?? "Signed in")}
            </span>
            <SignOutButton />
          </div>
        ) : (
          <Link href="/sign-in" className={buttonVariants({ variant: "outline", size: "sm" })}>
            Sign in
          </Link>
        )}
      </nav>
    </header>
  );
}
