from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_ids = []
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config
        self.is_incoming = True

        if self.dllm_config is not None:
            if len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.DECODE
            else:
                self.dllm_phase = DllmReqPhase.PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase == DllmReqPhase.PREFILL

    def determine_dllm_phase(self: Req):
        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.fill_ids) < min_required_length:
            # still incoming stage
            return

        input_block = self.fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.PREFILL
        else:
            self.dllm_phase = DllmReqPhase.DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if not self.dllm_ids:
            self.dllm_ids = (
                self.origin_input_ids
                + [self.dllm_config.mask_id] * self.dllm_config.block_size
            )
        else:
            self.dllm_block_offset += self.dllm_config.block_size
            self.dllm_ids += [self.dllm_config.mask_id] * self.dllm_config.block_size

        self.fill_ids = self.dllm_ids
