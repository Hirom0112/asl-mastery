# v3.0 docs archive (superseded by ADR 0011)

> Frozen 2026-05-21. Validation reports + the two ADRs that governed
> the pre-pivot machine-learning era, preserved as historical record.
> See [ADR 0011] for the pivot rationale.

[ADR 0011]: ../../decisions/0011-pivot-to-landmarks-templates.md

## Files

- **`0001-recognition-architecture.md`** — original Path B (end-to-end
  small 3D CNN). Superseded by ADR 0006 on 2026-05-19, reinstated by
  ADR 0010 on 2026-05-20, **superseded again by ADR 0011 on
  2026-05-21**. Live copy still at `docs/decisions/0001-...` (preserved
  with "Superseded by ADR 0011" header).
- **`0010-reversal-of-adr-0006.md`** — reversal of ADR 0006, reverted
  to ADR 0001 Path B on 2026-05-20. **Superseded by ADR 0011 on
  2026-05-21**. Live copy still at `docs/decisions/0010-...` (preserved
  with "Superseded by ADR 0011" header).
- **`v1.md`** — validation report for v1.0.1 (BiLSTM under the
  permissive ADR 0006 reading). **17.86 % top-1**, 36.31 % top-3 on
  80-sign vocabulary. Failed the eval gate; recorded honestly.
- **`v2.md`** — validation report for v2.0.0 (BiLSTM + ASL Citizen +
  Sem-Lex under ADR 0006). **67.07 % top-1**, 84.94 % top-3 on
  75-sign vocabulary. Below the 85 % gate; recorded honestly.
- **`v3.md`** — never-filled-in validation stub for the 3D-CNN run
  that ADR 0010 specified. The corresponding training never produced
  a checkpoint; preserved as the contract that didn't trigger.

Companion archive: `ml/archive/v3.0/` (artifacts, training code,
data manifests, run metadata).
