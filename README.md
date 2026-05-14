# SGLSim

This repository is being rebuilt as `sglsim`, a CPU-only simulator fork derived from SGLang.

The checkpointed SGLang-based project has been moved to `Archived/` and should be treated as migration reference material. New plans and new implementation files live outside `Archived/`.

## Supported

- Offline and blocking simulator modes
- AIConfigurator latency prediction
- CPU-only request scheduling and batching simulation
- Metadata-only KV/page/cache simulation
- Simulator metrics output
- Benchmark metrics loading

## Not Supported

- Real GPU inference
- CUDA graphs and CUDA kernels
- Distributed GPU execution
- Real model weight loading
- Real KV tensor transfer
- LoRA or checkpoint weight updates
