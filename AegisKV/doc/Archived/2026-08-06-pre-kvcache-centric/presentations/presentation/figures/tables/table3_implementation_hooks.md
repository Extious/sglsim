# Table 3: Implementation Hooks

| Hook | SGLang path | Purpose |
|---|---|---|
| request metadata | tokenizer manager / `Req` | attach workflow state |
| radix match/insert/split | `HiRadixCache` / `RadixCache` | update segment metadata and shared SafeCover |
| fixed Host D2H | `HiRadixCache.write_backup` / `HiCacheController.write` | observe fixed write-through copy |
| Host ACK | `HiRadixCache.writing_check` / `ack_write_queue` | publish HostBackup directory entry |
| H2D/cross-rank restore | `HiCacheController.load` / restore protocol | recover failed-rank Host frontier on retry rank |
| GPU eviction | `HiRadixCache.evict` | preserve/invalidate GPU hit state |
| Host eviction | `evict_host` | invalidate Host frontier |
| peer ACK | peer copy manager / DP controller | mark GPUReplica frontier |
| peer demotion | peer replica manager | invalidate or downgrade GPUReplica frontier |
| failure retry | scheduler / DP controller | choose frontier and retry path |
