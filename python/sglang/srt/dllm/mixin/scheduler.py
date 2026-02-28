from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Set, Union

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    CLIP_MAX_NEW_TOKENS,
    IGNORE_EOS_RESERVE_TOKENS,
    AddReqResult,
    PrefillAdderBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class DllmPrefillAdder(PrefillAdderBase):
    """
    Prefill adder for Diffusion LLM (DLLM) scheduling.

    This is a simplified version of PrefillAdder that:
    - Keeps core prefill adder functionality (budget management, token tracking)
    - Keeps DLLM-specific capabilities (block-based token allocation)
    - Keeps SWA (Sliding Window Attention) support
    - Removes hicache / priority scheduling / prefill_delayer / preempt features
    """

    def __init__(
        self,
        page_size: int,
        tree_cache: BasePrefixCache,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        dllm_config: Optional[DllmConfig] = None,
    ):
        super().__init__(
            page_size=page_size,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            running_batch=running_batch,
            new_token_ratio=new_token_ratio,
        )

        self.rem_input_tokens = rem_input_tokens
        self.dllm_config = dllm_config

        # Initialize DLLM-specific metadata
        if self.dllm_config is not None:
            self._init_dllm_meta(dllm_config)

        # For ignore_eos support: track token states of all requests
        self.req_states: Optional[List[tuple]] = None

        if running_batch is not None:
            self.rem_total_token_offset += sum(
                [
                    self._get_running_request_total_token_offset(r)
                    for r in running_batch.reqs
                ]
            )

    def _init_dllm_meta(self, dllm_config: DllmConfig):
        self.dllm_block_size = dllm_config.block_size
        max_running_reqs = dllm_config.max_running_requests
        self.rem_dllm_tokens = max_running_reqs * self.dllm_block_size

    def budget_state(self):
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0:
            return AddReqResult.OTHER

        if self.rem_dllm_tokens <= 0:
            return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _update_prefill_budget(
        self, prefix_len: int, extend_input_len: int, max_new_tokens: int
    ):
        extend_input_len = self.ceil_paged_tokens(extend_input_len)

        self.rem_total_token_offset += extend_input_len + max_new_tokens
        self.cur_rem_token_offset += extend_input_len
        self.rem_input_tokens -= extend_input_len

        if self.dllm_config is not None:
            self.rem_dllm_tokens -= extend_input_len

        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len

    def _get_dllm_remain_tokens(self) -> int:
        _rem_tokens = min(
            self.rem_dllm_tokens,
            self.dllm_block_size,
            int(self.rem_total_tokens),
        )
        if _rem_tokens <= 0:
            _rem_tokens = self.rem_dllm_tokens

        return _rem_tokens

    def _add_dllm_req(self, req: Req, prefix_len: int):
        # Make sure at least one page is available
        trunc_len = (
            min(self.rem_dllm_tokens, self.dllm_block_size)
            // self.page_size
            * self.page_size
        )

        req.extend_input_len = trunc_len
        req.fill_ids = req.fill_ids[: prefix_len + trunc_len]

        self.can_run_list.append(req)

        self._update_prefill_budget(prefix_len, trunc_len, 0)

    def add_req(self, req: Req):
        """
        - Incoming requests: Need full validation, locking, and initialization
        - Staging requests: Already have resources allocated, skip locking
        """
        # Staging requests: already have resources, skip locking
        if not req.is_incoming:
            return self._add_staging_req(req)

        # Incoming requests: full validation and locking required
        return self._add_incoming_req(req)

    def _add_staging_req(self, req: Req) -> AddReqResult:
        assert self.dllm_config is not None
        _rem_tokens = self._get_dllm_remain_tokens()

        if _rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        # Truncate input length to available tokens and update request metadata
        truncated = req.extend_input_len > _rem_tokens
        req.extend_input_len = min(req.extend_input_len, _rem_tokens)
        req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]
        self.can_run_list.append(req)

        # Update budget: reserve max_new_tokens only if not truncated
        max_new_tokens = (
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            if not truncated
            else 0
        )
        self._update_prefill_budget(0, req.extend_input_len, max_new_tokens)

        # Return based on remaining token availability
        return (
            AddReqResult.NO_TOKEN
            if self._get_dllm_remain_tokens() <= 0
            else AddReqResult.CONTINUE
        )

    def _add_incoming_req(self, req: Req) -> AddReqResult:
        # Handle ignore_eos case when tree cache is disabled
        if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
            return self.add_one_req_ignore_eos(req)

        total_tokens = req.extend_input_len + min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )

        input_tokens = self.ceil_paged_tokens(req.extend_input_len)
        prefix_len = len(req.prefix_indices)

        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN

        if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
            return AddReqResult.OTHER

        with self._lock_node(req.last_node):
            # self.rem_total_tokens may decrease after the lock acquisition
            if total_tokens >= self.rem_total_tokens:
                return AddReqResult.NO_TOKEN

            input_tokens = self.ceil_paged_tokens(req.extend_input_len)

            if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
                return AddReqResult.OTHER

            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER

            self._add_dllm_req(req, prefix_len)
            self._req_inc_lock_ref(req)

        return self.budget_state()

    def add_one_req_ignore_eos(self, req: Req):
        """
        Add one DLLM request with ignore_eos=True handling.

        This method tracks token states of all requests to ensure sufficient memory
        when ignore_eos is enabled (requests may generate unlimited tokens).
        """
        # Early exit if no enough tokens for the input tokens
        if self.ceil_paged_tokens(req.extend_input_len) > min(
            self.cur_rem_tokens, self.rem_total_tokens
        ):
            return AddReqResult.NO_TOKEN

        # FIXME: reduce redundant implementation
        def add_req_state(r, insert_sort=False):
            new_token_ratio = (
                1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
            )
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(
                r.output_ids
            )
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)

            if tokens_left <= 0:
                return

            if not insert_sort:
                self.req_states.append((tokens_left, tokens_occupied))
            else:
                i = 0
                for i in range(len(self.req_states)):
                    if tokens_left <= self.req_states[i][0]:
                        break
                self.req_states.insert(i, (tokens_left, tokens_occupied))

        if self.req_states is None:
            self.req_states = []
            add_req_state(req)
            if self.running_batch is not None:
                for r in self.running_batch.reqs:
                    add_req_state(r)
            for r in self.can_run_list:
                add_req_state(r)
            self.req_states.sort(key=lambda x: x[0])
        else:
            add_req_state(req, insert_sort=True)

        if not self.is_hybrid_swa:
            # Skip this logic for swa. The SWA has different memory management, and
            # this mechanism is underestimating the memory usage.
            cur_rem_tokens = self.cur_rem_tokens - self.ceil_paged_tokens(
                req.extend_input_len
            )
            tokens_freed = 0
            for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
                # tokens_left gives a reservative calculation as the last token is not stored
                bs = len(self.req_states) - i
                min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
                # reserve tokens for corner cases
                if min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs:
                    return AddReqResult.NO_TOKEN
                tokens_freed += tokens_occupied

        if self.rem_dllm_tokens <= 0:
            return AddReqResult.OTHER

        # For DLLM, use block-based allocation
        self._add_dllm_req(req, 0)
        self._req_inc_lock_ref(req)

        return self.budget_state()


class SchedulerDllmMixin:
    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if self.server_args.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)

    def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling."""
        if self.try_preemption:
            self.running_batch.batch_is_full = False

        # Early exit if batch is full or no requests available
        if self._should_skip_prefill():
            return None

        running_bs = len(self.running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        # Create prefill adder with resource constraints
        adder = self._create_dllm_prefill_adder(running_bs)

        # Initialize DLLM manager and transfer requests
        self.dllm_manager.init_next_round()
        self._fetch_waiting_reqs()

        # Process batches
        forward_mode = self._process_dllm_batches(adder)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        # Record metrics and update state
        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder, running_bs)

        # Create and prepare batch
        new_batch = self._create_dllm_batch(can_run_list, forward_mode)
        return new_batch

    def _fetch_waiting_reqs(self: Scheduler):
        # Calculate how many requests can be added to DLLM manager
        max_dllm_capacity = self.dllm_config.max_running_requests - len(
            self.dllm_manager.waiting_queue
        )
        num_requests_to_add = min(max_dllm_capacity, len(self.waiting_queue))

        if num_requests_to_add > 0:
            requests_to_add = self.waiting_queue[:num_requests_to_add]
            self.dllm_manager.add_waiting_reqs(requests_to_add)
            self.waiting_queue = self.waiting_queue[num_requests_to_add:]

    def _should_skip_prefill(self: Scheduler) -> bool:
        """Check if DLLM prefill should be skipped."""
        if (
            self.running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True

        running_bs = len(self.running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
        ):
            self.running_batch.batch_is_full = True
            return True

        return False

    def _create_dllm_prefill_adder(
        self: Scheduler, running_bs: int
    ) -> DllmPrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return DllmPrefillAdder(
            page_size=self.page_size,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            running_batch=self.running_batch,
            new_token_ratio=self.new_token_ratio,
            rem_input_tokens=self.max_prefill_tokens,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(self: Scheduler, adder: DllmPrefillAdder) -> ForwardMode:
        """Process prefill or decode batches for DLLM."""
        forward_mode = ForwardMode.DLLM_EXTEND

        # Try prefill batch first
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            self._process_dllm_batch(adder, prefill_reqs)
        else:
            # Fall back to decode batch
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_dllm_batch(adder, decode_reqs)

        return forward_mode

    def _process_dllm_batch(
        self,
        adder: DllmPrefillAdder,
        batch: List[Req],
    ) -> None:
        """Process a batch, separating staging and incoming requests using is_incoming."""
        # Process staging requests first (is_incoming=False)
        staging_reqs = [req for req in batch if not req.is_incoming]
        if staging_reqs:
            staging_result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if staging_result != AddReqResult.CONTINUE:
                return

        # Then process incoming requests (is_incoming=True)
        incoming_reqs = [req for req in batch if req.is_incoming]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(adder, incoming_reqs)

    def _update_state_for_batch(
        self: Scheduler,
        can_run_list: List[Req],
        adder: DllmPrefillAdder,
        running_bs: int,
    ) -> None:
        """Update state for the batch."""

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_chunked_count()

        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

    def _create_dllm_batch(
        self: Scheduler, can_run_list: List[Req], forward_mode: ForwardMode
    ) -> ScheduleBatch:
        """Create and prepare a new DLLM batch."""
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()
        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        # Record prefill stats for logging after forward
        from sglang.srt.observability.scheduler_metrics_mixin import PrefillStats

        new_batch.prefill_stats = PrefillStats(
            log_input_tokens=self.adder.log_input_tokens,
            log_hit_tokens=self.adder.log_hit_tokens,
            new_token_ratio=self.adder.new_token_ratio,
            running_bs=len(self.running_batch.reqs),
            num_new_seqs=len(can_run_list),
        )

        return new_batch

    def process_dllm_reqs(
        self: Scheduler,
        adder: DllmPrefillAdder,
        reqs: List[Req],
        check_batch_full: bool = False,
    ) -> AddReqResult:
        """Process DLLM requests with resource allocation.

        Args:
            adder: The prefill adder for resource management
            reqs: List of requests to process
            check_batch_full: If True, check batch capacity before adding each request
                              (used for incoming requests)
        """
        res = AddReqResult.CONTINUE
        for req in reqs:
            if check_batch_full:
                # Check if batch is full (only for incoming requests)
                running_bs = len(self.running_batch.reqs)
                if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                    self.running_batch.batch_is_full = True
                    break

                # Prepare incoming request
                req.init_next_round_input(self.tree_cache)

            res = adder.add_req(req)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN and check_batch_full:
                    self.running_batch.batch_is_full = True
                break

        return res

    def process_dllm_incoming_reqs(
        self: Scheduler, adder: DllmPrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation."""
        return self.process_dllm_reqs(adder, reqs, check_batch_full=True)

    def process_dllm_staging_reqs(
        self: Scheduler, adder: DllmPrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process staging DLLM requests with resource allocation."""
        return self.process_dllm_reqs(adder, reqs, check_batch_full=False)


class DllmManager:
    """
    Manager for Diffusion LLM request scheduling.

    Maintains two queues:
    - waiting_queue: The requests waiting to be scheduled with max running requests limit
    - staging_queue: Requests allocated resources by DllmPrefillAdder
    """

    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

    def get_prefill_requests(self) -> List[Req]:
        """Get all prefill requests from waiting queue."""
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]

    def get_decode_requests(self) -> List[Req]:
        """Get all decode requests from waiting queue."""
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to waiting queue with redundancy check."""
        assert self.dllm_config is not None, "Diffusion LLM config is not set."

        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]

        # Check for duplicate request IDs
        if self._has_duplicate_reqs(reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")

        self.waiting_queue.extend(reqs_to_add)

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to staging queue (allocated by DllmPrefillAdder)."""
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        """Check if any request ID already exists in waiting queue."""
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_staging_reqs(self) -> bool:
        """Check if there are requests in staging queue."""
        return self.dllm_config is not None and len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        """Check if both queues are empty or DLLM is not configured."""
        if self.dllm_config is None:
            return True
        return len(self.waiting_queue) == 0

    def increment_chunked_count(self) -> None:
        """Increment chunked count for all staging requests."""
        for req in self.staging_queue:
            req.is_chunked += 1

    def filter_finished_reqs(self) -> None:
        """Remove finished requests from both queues."""
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]

    def init_next_round(self) -> None:
        """Initialize staging requests for next round and clear staging queue."""
        for req in self.staging_queue:
            req.is_incoming = False  # Mark as staging request for next round
            req.init_next_round_input()
        self.staging_queue = []
