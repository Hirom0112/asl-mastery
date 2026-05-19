import Link from "next/link";

import { buttonVariants } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center bg-zinc-50 px-6 py-16 dark:bg-black">
      <main className="flex w-full max-w-md flex-col gap-4 text-center">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
          Page not found.
        </h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-300">
          That route does not exist. Try the home page or your dashboard.
        </p>
        <div className="flex justify-center gap-2">
          <Link href="/" className={buttonVariants()}>
            Home
          </Link>
          <Link href="/dashboard" className={buttonVariants({ variant: "outline" })}>
            Dashboard
          </Link>
        </div>
      </main>
    </div>
  );
}
