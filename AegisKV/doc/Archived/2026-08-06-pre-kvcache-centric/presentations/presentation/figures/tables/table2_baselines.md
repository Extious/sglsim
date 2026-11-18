# Table 2: Baselines

| Policy | Workflow | Lifecycle | Segment | Restore/recompute | Resource pressure |
|---|---|---|---|---|---|
| no backup | no | no | no | recompute | no |
| baseline write-through invisible | no | no | no | recompute | fixed D2H only |
| Host visible only | no | no | segment | restore/recompute | Host restore queue |
| uniform GPUReplica | no | no | segment | restore | peer budget only |
| request-level | weak | no | no | restore | coarse |
| agent-level | yes | weak | no | restore | coarse |
| static critical path | yes | weak | segment | restore | no |
| AegisKV | yes | yes | yes | yes | yes |
