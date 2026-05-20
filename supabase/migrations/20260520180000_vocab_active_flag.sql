-- Add `is_active_for_practice` to vocabulary_items. Practice scheduler
-- only prompts signs the v2.0.0 classifier can recognize (75 classes).
-- The other 21 signs in our 96-vocab list are out-of-model-scope and
-- would otherwise be served to learners and *always fail prediction*
-- (since the classifier outputs probs only over its 75 known classes).
--
-- 16 signs were dropped at the slice-1 ADR-0008 filter (insufficient
-- clip count). 5 more were dropped at 9c (`yes, blue, bad, happy,
-- woman` — bottom-five per-sign accuracy in v2-005).

alter table public.vocabulary_items
  add column if not exists is_active_for_practice boolean not null default true;

update public.vocabulary_items set is_active_for_practice = false
  where id in (
    'bad', 'big', 'blue', 'church', 'come', 'excuse', 'happy',
    'his', 'hungry', 'interpreter', 'love', 'our', 'thank_you',
    'they', 'we', 'what', 'when', 'where', 'why', 'woman', 'yes'
  );
