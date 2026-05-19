"use client";

import { Button } from "@/components/ui/button";

export default function ErrorBoundary({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center bg-zinc-50 px-6 py-16 dark:bg-black">
      <main className="flex w-full max-w-md flex-col gap-4 text-center">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
          Something went wrong.
        </h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-300">
          The page hit an unexpected error. Trying again often clears it.
        </p>
        {error.digest ? (
          <p className="text-[10px] text-zinc-400 dark:text-zinc-500">Error id {error.digest}</p>
        ) : null}
        <div className="flex justify-center">
          <Button onClick={() => reset()}>Try again</Button>
        </div>
      </main>
    </div>
  );
}
