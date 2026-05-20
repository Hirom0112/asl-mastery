"use client";

import { useState } from "react";
import Link from "next/link";

import type { VocabProgressItem } from "@/lib/scheduler/vocab-progression";
import styles from "./sidebar.module.css";

interface Props {
  items: VocabProgressItem[];
  currentSignId: string;
}

export function PracticeSidebar({ items, currentSignId }: Props) {
  const [open, setOpen] = useState(true);

  const mastered = items.filter((i) => i.state === "mastered").length;
  const total = items.length;

  return (
    <aside className={`${styles.sidebar} ${open ? styles.open : styles.closed}`}>
      <button
        type="button"
        className={styles.toggle}
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? "Collapse vocabulary list" : "Open vocabulary list"}
        aria-expanded={open}
      >
        {open ? "←" : "→"}
      </button>

      {open && (
        <div className={styles.body}>
          <div className={styles.header}>
            <p className={styles.headerEyebrow}>Vocabulary</p>
            <p className={styles.headerProgress}>
              <strong>{mastered}</strong>
              <span className={styles.progressOf}> of </span>
              <strong>{total}</strong>
              <span className={styles.progressOf}> mastered</span>
            </p>
            <div className={styles.progressBar} aria-hidden="true">
              <div
                className={styles.progressFill}
                style={{ width: `${(mastered / total) * 100}%` }}
              />
            </div>
          </div>

          <nav aria-label="Vocabulary progression" className={styles.list}>
            {items.map((item) => {
              const isCurrent = item.id === currentSignId;
              const baseClasses = `${styles.item} ${styles[`state_${item.state}`]} ${
                isCurrent ? styles.itemCurrent : ""
              }`;

              if (item.state === "locked") {
                return (
                  <div key={item.id} className={baseClasses} aria-disabled="true">
                    <span className={styles.itemIcon}>🔒</span>
                    <span className={styles.itemLabel}>{item.displayGloss}</span>
                  </div>
                );
              }

              return (
                <Link
                  key={item.id}
                  href={`/practice?sign=${encodeURIComponent(item.id)}`}
                  prefetch={false}
                  className={baseClasses}
                  aria-current={isCurrent ? "page" : undefined}
                >
                  <span className={styles.itemIcon}>
                    {item.state === "mastered" ? "✓" : isCurrent ? "●" : "○"}
                  </span>
                  <span className={styles.itemLabel}>{item.displayGloss}</span>
                </Link>
              );
            })}
          </nav>

          <p className={styles.footnote}>Unlock the next sign by mastering this one.</p>
        </div>
      )}
    </aside>
  );
}
