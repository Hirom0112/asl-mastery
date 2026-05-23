"use client";

import { useState } from "react";
import Link from "next/link";

import type { VocabProgressItem, VocabProgressState } from "@/lib/scheduler/vocab-progression";
import styles from "./sidebar.module.css";

interface Props {
  items: VocabProgressItem[];
  currentSignId: string;
  greetingName?: string;
}

interface Group {
  category: string;
  items: VocabProgressItem[];
}

// Explicit display names for category slugs. Anything not listed falls back
// to title-casing the slug, so new categories still render reasonably.
const CATEGORY_LABELS: Record<string, string> = {
  "wh-question": "Question Words",
  "greetings-social": "Greetings & Social",
  "yes-no": "Yes & No",
  "verbs-common": "Common Verbs",
  "deaf-culture": "Signing & Identity",
  descriptors: "Describing Words",
  food: "Food & Drink",
};

function prettyCategory(c: string): string {
  return (
    CATEGORY_LABELS[c] ??
    c
      .split("-")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ")
  );
}

// Group items into categories. Items keep difficulty-rank order within a
// category (the input is already globally rank-ordered); categories are then
// sorted easiest → hardest by their average difficulty rank.
function groupByCategory(items: VocabProgressItem[]): Group[] {
  const groups: Group[] = [];
  const index = new Map<string, number>();
  for (const item of items) {
    let i = index.get(item.category);
    if (i === undefined) {
      i = groups.length;
      index.set(item.category, i);
      groups.push({ category: item.category, items: [] });
    }
    groups[i].items.push(item);
  }

  const avgRank = (g: Group) => {
    const ranks = g.items.map((it) => it.difficultyRank).filter((r): r is number => r != null);
    return ranks.length
      ? ranks.reduce((a, b) => a + b, 0) / ranks.length
      : Number.POSITIVE_INFINITY;
  };
  groups.sort((a, b) => avgRank(a) - avgRank(b));
  return groups;
}

function StateIndicator({ state }: { state: VocabProgressState }) {
  if (state === "locked") {
    return (
      <svg
        className={styles.lockIcon}
        viewBox="0 0 24 24"
        width="11"
        height="11"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        aria-hidden="true"
      >
        <rect x="5" y="11" width="14" height="9" rx="2.5" />
        <path d="M8 11V8a4 4 0 0 1 8 0v3" />
      </svg>
    );
  }
  if (state === "mastered") {
    return (
      <svg
        className={styles.checkIcon}
        viewBox="0 0 24 24"
        width="12"
        height="12"
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d="M5 13l4 4L19 7" />
      </svg>
    );
  }
  // current = filled terracotta dot; unlocked = hollow ring
  return (
    <span className={state === "current" ? styles.dotFilled : styles.dotRing} aria-hidden="true" />
  );
}

export function PracticeSidebar({ items, currentSignId, greetingName }: Props) {
  const [open, setOpen] = useState(true);

  const groups = groupByCategory(items);
  const mastered = items.filter((i) => i.state === "mastered").length;
  const total = items.length;

  // Open the category that holds the active sign by default; collapse the rest.
  const [openCats, setOpenCats] = useState<Record<string, boolean>>(() => {
    const active =
      items.find((i) => i.id === currentSignId) ?? items.find((i) => i.state === "current");
    return active ? { [active.category]: true } : {};
  });

  const toggleCat = (category: string) =>
    setOpenCats((prev) => ({ ...prev, [category]: !prev[category] }));

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
            {greetingName ? <p className={styles.greeting}>Hi, {greetingName}</p> : null}
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
                style={{ width: `${total ? (mastered / total) * 100 : 0}%` }}
              />
            </div>
          </div>

          <nav aria-label="Vocabulary progression" className={styles.list}>
            {groups.map((group) => {
              const isOpen = !!openCats[group.category];
              const groupMastered = group.items.filter((i) => i.state === "mastered").length;
              return (
                <div key={group.category} className={styles.category}>
                  <button
                    type="button"
                    className={styles.categoryHeader}
                    onClick={() => toggleCat(group.category)}
                    aria-expanded={isOpen}
                  >
                    <svg
                      className={`${styles.chevron} ${isOpen ? styles.chevronOpen : ""}`}
                      viewBox="0 0 24 24"
                      width="12"
                      height="12"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.4"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <path d="M9 6l6 6-6 6" />
                    </svg>
                    <span className={styles.categoryName}>{prettyCategory(group.category)}</span>
                    <span className={styles.categoryMeta}>
                      {groupMastered}/{group.items.length}
                    </span>
                  </button>

                  <div
                    className={`${styles.categoryItems} ${isOpen ? styles.categoryItemsOpen : ""}`}
                  >
                    <div className={styles.categoryItemsInner}>
                      {group.items.map((item) => {
                        const isCurrent = item.id === currentSignId;
                        const baseClasses = `${styles.item} ${styles[`state_${item.state}`]} ${
                          isCurrent ? styles.itemCurrent : ""
                        }`;

                        if (item.state === "locked") {
                          return (
                            <div key={item.id} className={baseClasses} aria-disabled="true">
                              <span className={styles.itemIcon}>
                                <StateIndicator state={item.state} />
                              </span>
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
                              <StateIndicator state={item.state} />
                            </span>
                            <span className={styles.itemLabel}>{item.displayGloss}</span>
                          </Link>
                        );
                      })}
                    </div>
                  </div>
                </div>
              );
            })}
          </nav>

          <p className={styles.footnote}>Unlock the next sign by mastering this one.</p>
        </div>
      )}
    </aside>
  );
}
