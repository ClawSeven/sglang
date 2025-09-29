from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np

from sglang.srt.diffusion.algorithm.base import DiffusionAlgorithm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from typing import List

class LowConfidence(DiffusionAlgorithm):

    def run(self,
            model_runner: ModelRunner,
            forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], Optional[List[int]], bool
    ]:
        
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [diffusion_block_size * batch_size, hidden_size]

        # fixme: add quick path 
        start_list = []
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = (block_input_ids == 156895)

            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        for _ in range(self.block_size):
            mask_index = (forward_batch.input_ids == 156895)
            if torch.sum(mask_index).item() == 0:
                break

            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None, save_kv_cache=False
            )

            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size

            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = (block_input_ids == 156895)
                if torch.sum(block_mask_index).item() == 0:
                    continue
                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(torch.gather(
                            F.softmax(curr_logits, dim=-1), dim=-1,
                            index=torch.unsqueeze(x, -1)), -1)
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)
                transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                _, select_index = torch.topk(confidence, k=1)
                transfer_index[select_index] = True
                
                block_input_ids[transfer_index] = x[transfer_index]

        logits_output, can_run_cuda_graph = model_runner.forward(
            forward_batch, pp_proxy_tensors=None, save_kv_cache=True
        )

        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
    
        if not start_list:
            start_list = None

        return logits_output, next_token_ids, start_list, can_run_cuda_graph

Algorithm = LowConfidence
