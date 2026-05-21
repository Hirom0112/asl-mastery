-- Deactivate all model_versions rows.
--
-- The earlier permissive clarification on pretrained landmark
-- detectors was withdrawn on 2026-05-20. Brief Requirement 7's
-- strict reading governs again: no pretrained vision components
-- anywhere in the pipeline. The existing v1.0.1, v2.0.0, and
-- v2.1.0 artifacts were all trained on MediaPipe Holistic landmark
-- features and are no longer compliant. ADR 0010 documents the
-- reversal.
--
-- After this migration, getActiveModelVersion() returns null and the
-- practice screen falls back to stubPredict(). This is the honest
-- production state until v3.x is rebuilt under the new constraint.

update public.model_versions set is_active = false where is_active = true;
