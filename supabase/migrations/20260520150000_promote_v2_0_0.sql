-- Promote v2.0.0 to the active model.
--
-- v2.0.0 lifts top-1 from v1.0.1's 17.86% → 67.07% on 75 signs
-- (top-3: 84.94%). Still below the 85% eval-gate floor; promoted under
-- the slice-1 acceptance pattern documented in docs/validation/v2.md.
-- See ADR 0009 for the ASL Citizen / MSR-LA disclosure.

-- Deactivate any prior active row (the unique-active-row index would
-- otherwise reject the insert/update below).
update public.model_versions set is_active = false where is_active = true;

insert into public.model_versions (
  id,
  artifact_url,
  config_url,
  metrics_url,
  is_active,
  promoted_at
) values (
  'v2.0.0',
  'https://pub-1d2c66f9b93e4a00a257a3dc0675d73b.r2.dev/v2.0.0/classifier.onnx',
  'https://pub-1d2c66f9b93e4a00a257a3dc0675d73b.r2.dev/v2.0.0/config.json',
  'https://pub-1d2c66f9b93e4a00a257a3dc0675d73b.r2.dev/v2.0.0/validation.json',
  true,
  now()
)
on conflict (id) do update set
  is_active   = excluded.is_active,
  promoted_at = excluded.promoted_at;
