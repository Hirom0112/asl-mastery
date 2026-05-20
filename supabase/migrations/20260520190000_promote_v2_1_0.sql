-- Promote v2.1.0 to the active model.
--
-- v2-009 lifts top-1 from v2-007's 50.25% → 64.10% (+13.84pp) on the
-- same v2f test set (apples-to-apples; v2-007's prior 67.07% number
-- was inflated by a tiny 498-clip test set with 13 signs at 100% on
-- n≤10). Top-3: 78.45% vs v2-007's 68.39% (+10.06pp). Significant.
--
-- Built from the 3-fix deep-dive findings:
--   1. Variant cleanup (drop eat_1/eat_2 collisions, fix mislabels)
--   2. mediapipe_noise_apply default-on (kills source-mask shortcut)
--   3. --epochs 40 / T_max 40 (LR schedule aligned with peak)
--
-- Still below 85% eval-gate floor; slice-1 acceptance pattern continues.

update public.model_versions set is_active = false where is_active = true;

insert into public.model_versions (
  id, artifact_url, config_url, metrics_url, is_active, promoted_at
) values (
  'v2.1.0',
  'https://pub-1d2c66f9b93e4a00a257a3dc0675d73b.r2.dev/v2.1.0/classifier.onnx',
  'https://pub-1d2c66f9b93e4a00a257a3dc0675d73b.r2.dev/v2.1.0/config.json',
  'https://pub-1d2c66f9b93e4a00a257a3dc0675d73b.r2.dev/v2.1.0/validation.json',
  true,
  now()
)
on conflict (id) do update set
  is_active = excluded.is_active,
  promoted_at = excluded.promoted_at;
