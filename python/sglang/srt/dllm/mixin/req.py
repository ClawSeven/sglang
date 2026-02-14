from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


# TODO(clawseven): Refactor the ambiguous enum names once PrefillAdder
# unifies the execution flow of incoming and staging requests
class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    # Represents partially decoded dLLM decoding requests that have not yet completed
    STAGING_DECODE_PARTIAL = "staging_decode_partial"

    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_ids = []
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config
        self.dllm_start_offset = 0

        if self.dllm_config is not None:
            if len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.fill_ids) < min_required_length:
            # still incoming stage
            return

        input_block = self.fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def skip_initialization(self: Req) -> bool:
        # Skip initialization for staging decode partial phase since it can be called multiple times
        return self.dllm_phase == DllmReqPhase.STAGING_DECODE_PARTIAL

    def recover_phase(self: Req):
        if self.dllm_phase == DllmReqPhase.STAGING_DECODE_PARTIAL:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def mark_partial_decode(self: Req):
        if self.dllm_phase == DllmReqPhase.STAGING_DECODE:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE_PARTIAL

    def _init_fill_ids_for_dllm(self: Req):
        if not self.skip_initialization():
            if not self.dllm_ids:
                self.dllm_ids = (
                    self.origin_input_ids
                    + [self.dllm_config.mask_id] * self.dllm_config.block_size
                )
            else:
                self.dllm_block_offset += self.dllm_config.block_size
                self.dllm_ids += [
                    self.dllm_config.mask_id
                ] * self.dllm_config.block_size

        self.fill_ids = self.dllm_ids

    def determine_dllm_start_offset(self: Req):
        block_size = self.dllm_config.block_size
        input_len = len(self.origin_input_ids)

        # Calculate tokens that don't form a complete block at the start
        offset_len = input_len % block_size

        # Only set offset if we have prefix tokens and haven't processed blocks yet
        # dllm_block_offset tracks cumulative processed token count
        if (
            offset_len > 0
            and self.dllm_block_offset == (input_len // block_size) * block_size
        ):
            self.dllm_start_offset = offset_len
        else:
            self.dllm_start_offset = 0
