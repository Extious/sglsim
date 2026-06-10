from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglsim.srt.managers.prefill_adder import PrefillAdder
    from sglsim.srt.managers.schedule_batch import Req, ScheduleBatch


@dataclasses.dataclass
class PrefillStats:
    log_input_tokens: int
    log_hit_tokens: int
    new_token_ratio: float
    num_running_reqs: int
    num_new_seqs: int

    @classmethod
    def from_adder(cls, adder: PrefillAdder, running_reqs: list[Req]) -> PrefillStats:
        return cls(
            log_input_tokens=adder.log_input_tokens,
            log_hit_tokens=adder.log_hit_tokens,
            new_token_ratio=adder.new_token_ratio,
            num_running_reqs=len(running_reqs),
            num_new_seqs=len(adder.can_run_list),
        )


class SchedulerMetricsMixin:
    def init_metrics(self) -> None:
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.decode_log_sim_dur = 0.0
        self.last_prefill_tokens = 0
        self.last_gen_throughput = 0.0
        self.last_input_throughput = 0.0

    def report_prefill_stats(self, batch: ScheduleBatch, prefill_stats: PrefillStats) -> None:
        sim_dur = max(batch.inference_dur_s, 0.0)
        self.last_input_throughput = (
            prefill_stats.log_input_tokens / sim_dur if sim_dur > 0 else 0.0
        )
        self.last_prefill_tokens = prefill_stats.log_input_tokens

        logical_tokens = self.cache.logical_kv_tokens(self.running_batch.reqs)
        used_tokens, total_tokens, avail_tokens, evictable_tokens = (
            self.cache.kv_token_usage_details(logical_tokens=logical_tokens)
        )
        token_usage = used_tokens / total_tokens if total_tokens > 0 else 0.0
        if os.getenv("SGLSIM_LOG_PROFILE", "").lower() == "sglang_projected":
            self.logger.info(
                "Prefill batch, "
                "#new-seq: %d, "
                "#new-token: %d, "
                "#cached-token: %d, "
                "token usage: %.2f, "
                "#running-req: %d, "
                "#queue-req: %d, "
                "cuda graph: True, "
                "input throughput (token/s): %.2f",
                prefill_stats.num_new_seqs,
                prefill_stats.log_input_tokens,
                prefill_stats.log_hit_tokens,
                token_usage,
                prefill_stats.num_running_reqs,
                len(self.waiting_queue),
                self.last_input_throughput,
            )
            return
        self.logger.info(
            "Prefill batch, "
            "#new-seq: %d, "
            "#new-token: %d, "
            "#cached-token: %d, "
            "token usage: %.2f, "
            "#running-req: %d, "
            "#queue-req: %d, "
            "#logical-token: %d, "
            "#avail-token: %d, "
            "#evictable-token: %d, "
            "input throughput (token/s): %.2f",
            prefill_stats.num_new_seqs,
            prefill_stats.log_input_tokens,
            prefill_stats.log_hit_tokens,
            token_usage,
            prefill_stats.num_running_reqs,
            len(self.waiting_queue),
            logical_tokens,
            avail_tokens,
            evictable_tokens,
            self.last_input_throughput,
        )

    def report_decode_stats(self, batch: ScheduleBatch) -> None:
        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        self.num_generated_tokens += len(batch.reqs)
        self.decode_log_sim_dur += max(batch.inference_dur_s, 0.0)

        interval = max(int(self.sched_cfg.decode_log_interval), 1)
        if self.forward_ct_decode % interval != 0:
            return

        self.last_gen_throughput = (
            self.num_generated_tokens / self.decode_log_sim_dur
            if self.decode_log_sim_dur > 0
            else 0.0
        )
        self.num_generated_tokens = 0
        self.decode_log_sim_dur = 0.0

        logical_tokens = self.cache.logical_kv_tokens(batch.reqs)
        used_tokens, total_tokens, avail_tokens, evictable_tokens = (
            self.cache.kv_token_usage_details(logical_tokens=logical_tokens)
        )
        token_usage = used_tokens / total_tokens if total_tokens > 0 else 0.0
        if os.getenv("SGLSIM_LOG_PROFILE", "").lower() == "sglang_projected":
            self.logger.info(
                "Decode batch, "
                "#running-req: %d, "
                "#token: %d, "
                "token usage: %.2f, "
                "cuda graph: True, "
                "gen throughput (token/s): %.2f, "
                "#queue-req: %d",
                len(batch.reqs),
                used_tokens,
                token_usage,
                self.last_gen_throughput,
                len(self.waiting_queue),
            )
            return
        self.logger.info(
            "Decode batch [%d], "
            "#running-req: %d, "
            "#token: %d, "
            "token usage: %.2f, "
            "gen throughput (token/s): %.2f, "
            "#queue-req: %d, "
            "#logical-token: %d, "
            "#avail-token: %d, "
            "#evictable-token: %d",
            self.forward_ct_decode,
            len(batch.reqs),
            used_tokens,
            token_usage,
            self.last_gen_throughput,
            len(self.waiting_queue),
            logical_tokens,
            avail_tokens,
            evictable_tokens,
        )
