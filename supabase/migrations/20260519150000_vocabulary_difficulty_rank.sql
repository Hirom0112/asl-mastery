-- Per-sign difficulty rank: a 1-N ordering from easiest to hardest.
--
-- Used by the scheduler to introduce untouched signs in a sensible
-- order — beginners shouldn't first hit "INTERPRETER" (high
-- phonological complexity + L13 Lifeprint lesson). With this rank in
-- place, `getNextItem()` prefers lower-ranked items when introducing
-- new content.
--
-- Populated in the next migration. Defaults to NULL so signs with no
-- rank fall to the end of the order naturally.

alter table public.vocabulary_items
  add column difficulty_rank integer;

-- Allow ascending sort on rank with NULLs last.
create index vocabulary_items_difficulty_rank
  on public.vocabulary_items (difficulty_rank nulls last);
