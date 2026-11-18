# AegisKV SGLSim figure captions

All panels use frozen evidence and are recommended as EuroSys `figure*`
floats at `\textwidth`.

- **S1**: Modeled WCT distributions for six architectures and two failure
  domains. Worker fault gains are dominated by avoiding the shared 60 s Native
  replacement; node faults also lose local replicas. Ten paired seeds;
  A100-40GiB metadata DES.
- **S2**: AegisKV WCT reduction versus Native Mooncake over concurrency and
  failure rate. Benefits peak at moderate load and collapse at C=8/16 as
  pressure reaches 100 percent and Peer-HBM protection approaches zero. Ten
  paired seeds.
- **S3**: Modeled scale and context sensitivity for Qwen3-8B TP1/PP1 and
  Qwen3-32B TP1/PP4. Fixed probes per cell dilute all-workflow gains as DP
  grows; the 32B service profile and failover delays are analytic rather than
  A100 measurements.
- **S4**: Algorithms 1-3 mechanism ablations. Slack priority,
  independent-tier outcomes, and AIMD behavior separate cleanly, while paired
  end-to-end WCT effects remain weak or negative. All three paired seeds and
  confidence intervals are retained.
- **S5**: AIMD and occupancy under low, high, and recovery phases. Alpha
  contracts to about 0.037 and later recovers to about 0.125; residual pressure
  explains the nonmonotonic windows. Three-seed A100-40GiB metadata DES.
- **S6**: Shared-prefix reuse mechanisms. Stable content identity and per-tier
  epochs reduce physical copies while retaining more reusable data, so lower
  copy amplification and higher occupancy are compatible. Sixty modeled
  cells, three paired seeds.
- **S7**: Directional bandwidth and repair sensitivity. Asynchronous off-path
  copies make bandwidth effects nearly flat, whereas modeled replacement time
  drives repair sensitivity. All 168 cells and 2,016 frozen failure-target
  probes are retained.
- **S8**: Six-architecture comparison under Worker/GPU and source-node
  failures. The node fault removes same-node replicas and yields much smaller
  gains. This view is derived from the same ten-seed sample as S1, not an
  independent replication.
- **S9**: Independent RTX4090 calibration holdout. Curves fit only on job
  141298 achieve 3.72 percent overall MAPE on all 80 job-141411 samples; 15 of
  16 cells meet 10 percent, while 64 MiB Peer-HBM recovery reaches 10.64
  percent. No refit or exclusion.
