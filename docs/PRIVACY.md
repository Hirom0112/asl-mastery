# Privacy

> What data this system collects, what it does not collect, and the
> architectural mechanisms that enforce the difference.

---

## 1. Core commitments

1. **Video stays on the learner's device** during normal practice
   sessions. The recognition model runs in the browser. Frames are
   never uploaded.
2. **Only metadata leaves the device.** When a learner records an
   attempt, what reaches the backend is: the prompted sign id, the
   predicted class id, the confidence value, the pass/fail decision,
   the timestamp, the model version id. Not the frames. Not the audio.
   Not the canvas data.
3. **Contributor video — the recording tool path — is the only
   exception**, and only under explicit, signed consent before the
   contributor is granted access to the admin-gated tool.

---

## 2. Architectural enforcement

- The learner app does not contain any code path that sends a video
  blob, canvas data, or frame tensor to the backend. This is enforceable
  by code review and by network-traffic audit; both should be done
  before each release.
- Sentry is configured with `replaysOnErrorSampleRate: 0` or with
  `Replay({ maskAllInputs: true, blockAllMedia: true })` to ensure
  camera streams are never captured in error replays.
- PostHog is configured with the canvas element carrying a `no-track`
  class and the corresponding ignore rule in the PostHog config.
- The `raw-training-data/` R2 bucket is private. The training-pipeline
  service has read access via a credential that is never present in
  the client bundle.
- The `references/` bucket (canonical reference videos; slice 1 sources from WLASL/MS-ASL with attribution per ADR 0004, slice 2 replaces with instructor-recorded clips) is
  public-read, since these are intentionally public reference content.

---

## 3. What we collect from learners

| Data | Where it lives | Why we have it |
|---|---|---|
| Email address | Postgres `users` table | Auth. |
| Handedness | Postgres `users` table | Inference-time frame flipping for left-handed learners. |
| Fitzpatrick scale (optional) | Postgres `users` table | Per-demographic accuracy reporting in eval gate. Affirmative opt-in only. |
| Attempts (metadata only) | Postgres `attempts` table | Mastery state, hint efficacy, eval signals. No frames. |
| Mastery state | Postgres `mastery_state` table | The pedagogical core of the product. |
| Learner-disagreed flags | Postgres `attempts` table | Quality signal for retraining. |

We do not collect: video, audio, screenshots, biometric identifiers,
device fingerprints beyond what auth requires, location data,
contacts, calendars.

---

## 4. What we collect from contributors (recording tool path)

This is the consent-required path.

| Data | Why |
|---|---|
| Video clips | Training data. |
| Signer id, sign id, take number | Provenance and dedup. |
| Lighting / background / sleeve / handedness | Augmentation and fairness analysis. |
| Fitzpatrick scale (optional) | Per-demographic data for fairness evaluation. |
| Self-rated correctness | Quality signal during cleaning. |
| Consent record | Proof we are authorized to use these clips. |

Each contributor signs a consent form before the recording tool is
unlocked for them. The form names: what is being recorded, where it is
stored, who has access, how it will be used (training, fairness
analysis), the retention policy, and the contributor's right to
revoke. Revocation triggers deletion of clips from R2 and from any
training manifest that has not yet been used.

---

## 5. Retention

- **Learners:** all data is retained as long as the account exists.
  Account deletion is a single user-initiated action that:
  1. Deletes the `users` row (cascades to `attempts` and `mastery_state`).
  2. Logs the deletion (timestamp, hashed identifier only) for audit.
- **Contributors:** clips are retained as long as the contributor has
  not revoked consent. Revocation deletes the clips from R2 and from
  future training data manifests. Already-trained models are not
  re-trained on revocation — clips that influenced a deployed model
  cannot be retroactively unlearned, and contributors are told this
  in the consent form.

---

## 6. Third parties

| Service | Data sent | Mitigation |
|---|---|---|
| Supabase | Email, handedness, attempts metadata | Standard data-processing terms. |
| Vercel | App logs, metadata | No video; canvas data is not sent. |
| Cloudflare R2 | Model artifacts (public); reference videos (public); raw training data (private). | Private bucket has restricted IAM. |
| Sentry | Error data | Replays disabled or fully masked. |
| PostHog | Product events with explicit, named field allowlist | No DOM autocapture on the practice screen. |

No advertising networks. No cross-site tracking. No data brokers.

---

## 7. What we do not promise

- We do not promise the system is anonymous to law enforcement subpoenas
  served on Supabase or Vercel. Standard data-processing legal regimes
  apply.
- We do not promise the model is unbiased; we promise to *measure*
  bias and *report* it in the validation report.
- We do not promise that disabling consent rescues clips already used
  to train a deployed model.

Honesty about what we are not promising is part of the trust contract.

---

## 8. How a learner can verify these claims

A learner can:

- Open browser DevTools, navigate to the Network panel, record a
  practice session, and observe that no video data leaves the device.
  Only JSON payloads with the fields named in Section 3.
- Inspect the page source and confirm the model artifact is served
  from R2 to the browser, not the other way around.
- Request a data export at any time; the export contains every row
  in `users`, `attempts`, and `mastery_state` keyed to their account.
- Request account deletion at any time and verify (via re-export
  attempt failing) that data is gone.

The transparency tools are part of the product, not afterthoughts.
