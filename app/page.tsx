export default function Home() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center bg-zinc-50 px-8 py-16 dark:bg-black">
      <main className="flex w-full max-w-2xl flex-col gap-10">
        <header className="flex flex-col gap-3">
          <p className="text-xs font-medium tracking-widest text-zinc-500 uppercase dark:text-zinc-400">
            ASL Mastery — pilot scaffolding
          </p>
          <h1 className="text-4xl font-semibold tracking-tight text-zinc-950 sm:text-5xl dark:text-zinc-50">
            A mastery-based skill acquisition system.
          </h1>
          <p className="max-w-xl text-lg leading-relaxed text-zinc-600 dark:text-zinc-300">
            ASL vocabulary is the controlled testbed. The unit of progress is the sign-mastered, not
            the minute-spent. Local inference, eval-gated quality, self-paced exit on mastery.
          </p>
        </header>

        <section className="grid gap-3 text-sm text-zinc-700 dark:text-zinc-300">
          <p className="font-medium text-zinc-900 dark:text-zinc-100">
            This is Phase 2 scaffolding. The learner experience is not here yet.
          </p>
          <ul className="grid gap-1 text-zinc-600 dark:text-zinc-400">
            <li>Architecture: see docs/ARCHITECTURE.md in the repo.</li>
            <li>Recognition path: landmark-based (ADR 0006).</li>
            <li>Pedagogical theory: docs/PEDAGOGY.md with verified citations.</li>
            <li>Vocabulary (96 signs): docs/VOCABULARY.md.</li>
            <li>Eval gate: docs/EVAL_GATE.md.</li>
          </ul>
        </section>

        <footer className="flex flex-col gap-2 border-t border-zinc-200 pt-6 text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
          <p>Pilot built for Superbuilders. Public-sources-only sourcing per ADR 0004.</p>
          <p>
            Next: Phase 3 — recording tool and dataset assembly. Phase 4 — landmark-classifier
            training.
          </p>
        </footer>
      </main>
    </div>
  );
}
