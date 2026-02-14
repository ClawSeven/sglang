from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        mask_index = forward_batch.input_ids == self.mask_id

        # Prefill path, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph

        # Decode path
        start_list = forward_batch.dllm_start_offsets

        # Track which blocks still have masked tokens (incomplete)
        # True = block still has work to do, False = block fully decoded
        incomplete_blocks = [True] * batch_size

        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size

            # Unmask tokens for each block
            for batch_id in range(batch_size):
                # Skip already completed blocks
                if not incomplete_blocks[batch_id]:
                    continue

                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id

                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)

                transfer_index = confidence > self.threshold

                if transfer_index.sum().item() == 0:
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                block_input_ids[transfer_index] = x[transfer_index]

            # Check if any block completed after unmasking
            any_block_completed = False
            for batch_id in range(batch_size):
                if not incomplete_blocks[batch_id]:
                    continue

                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id

                if torch.sum(block_mask_index).item() == 0:
                    incomplete_blocks[batch_id] = False
                    any_block_completed = True

            # TODO(clawseven): Add some strategy for better utilization
            # Early exit if batch_size > 1 and any block just completed
            if batch_size > 1 and any_block_completed:
                break

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = []

        is_early_exit = batch_size > 1 and not all(incomplete_blocks)

        for i in range(batch_size):
            if is_early_exit and incomplete_blocks[i]:
                # For blocks still incomplete in early exit, return empty tensor
                next_token_ids_list.append(
                    torch.tensor(
                        [], dtype=next_token_ids.dtype, device=next_token_ids.device
                    )
                )
            else:
                # For completed blocks or single batch (no early exit), return tokens from start position
                next_token_ids_list.append(next_token_ids[i, start_list[i] :])

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
