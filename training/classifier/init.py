"""Weight initialization — the visible audit surface for the
'no-pretrained-pipeline' claim per docs/MODEL.md §7.

A reviewer auditing the no-pretrained claim can read this file and
confirm:
- Every linear and projection layer is initialized via
  `torch.nn.init.kaiming_normal_` (the Kaiming/He initialization).
- LSTM internal weights use PyTorch's defaults (orthogonal /
  xavier), as documented in docs/MODEL.md §1.
- Biases are zeroed.
- No `load_state_dict` call anywhere in this file or in `model.py`.
- No external weight URL anywhere in this file or in `model.py`.

Programming frameworks (PyTorch, ONNX) are permitted by the brief
(Requirement 6). MediaPipe is permitted under ADR 0006 (general-
purpose landmark detector, used as a library). No pretrained ASL
or sign-classifier weights enter the system.
"""

from __future__ import annotations

import torch
from torch import nn


def init_classifier_weights(model: nn.Module) -> None:
    """Apply from-scratch Kaiming initialization to the classifier."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            # LSTM internals use PyTorch defaults (orthogonal/xavier)
            # per docs/MODEL.md §1. Zero the biases.
            for name, param in module.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                nn.init.kaiming_normal_(module.in_proj_weight, nonlinearity="relu")
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            nn.init.kaiming_normal_(module.out_proj.weight, nonlinearity="relu")
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)


# Repository invariant: this module must never call the pretrained-
# weights load API (i.e. `.load_state_dict(...)`). Audit surface is
# documented at docs/MODEL.md §7. The previously-attempted module-
# level string check was broken (it self-matched on its own error
# message). The eval-gate enforcer in scripts/check_eval_gate.py and
# the validation report's model_architecture field carry the
# audit trail forward.
