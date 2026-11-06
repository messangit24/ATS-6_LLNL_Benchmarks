# LLNL ATS-6 Benchmark Suite on Google Cloud TPU v6e (Trillium)

Unified 16-Chip (`4x4` 2D Torus) implementation and hardware telemetry suite for the **Lawrence Livermore National Laboratory (LLNL) Advanced Technology System 6 (ATS-6)** benchmarks on **Google Cloud TPU v6e (Trillium)** running on Google Kubernetes Engine (GKE).

---

## Repository Structure

| File | Description |
|---|---|
| [`ats6_pallas_benchmarks.py`](./ats6_pallas_benchmarks.py) | Unified 16-chip JAX / Pallas / `shard_map` benchmark runner covering all 9 LLNL ATS-6 workloads with on-device initialization and 8-subsystem XProf hardware telemetry (`HBM`, `VMEM`, `SMEM`, `SPMEM`, `VPU/XLU`, `MXU`, `DMA`, `ICI`). |
| [`ats6_tpuc_kernels.cc`](./ats6_tpuc_kernels.cc) | Native TPU-C (`tpucc` C++20) device kernels implementing asynchronous software-pipelined ping-pong double buffering (`256 KiB` DMA bursts across hardware `__sflag` registers) and `#pragma unroll 8` VREG loops. |
| [`ats6_tpuc_benchmarks.py`](./ats6_tpuc_benchmarks.py) | Standalone ahead-of-time (`tpurt`) benchmark harness for the Native TPU-C C++20 device kernels targeting `-mtpu=v6e`. |
| [`gke_tpu_v6e_job.yaml`](./gke_tpu_v6e_job.yaml) | GKE Multi-Host 16-Chip (`4x4` topology across 4 `ct6e-standard-4t` nodes) Indexed Job and Headless Service manifest. |
| [`ats6_benchmarks_overview_and_results.md`](./ats6_benchmarks_overview_and_results.md) | Detailed breakdown of what was benchmarked across all 9 workloads, their scientific/HPC/AI use cases, and complete 16-chip hardware telemetry tables. |
| [`ats6_xprof_summary.md`](./ats6_xprof_summary.md) | Markdown hardware telemetry summary generated directly from the 16-chip Cloud TPU v6e GKE run. |
| [`ats6_xprof_summary.json`](./ats6_xprof_summary.json) | Raw JSON telemetry output for all 9 workloads across all 8 TPU v6e hardware subsystems. |
| [`ats6_gke_benchmark_output.log`](./ats6_gke_benchmark_output.log) | Full execution log from the 4-worker (16-chip) GKE benchmark job. |

---

## Cloud TPU v6e (`4x4` = 16 Chips) Architecture Specifications

| Subsystem | Per-Chip Specification | 16-Chip (`4x4` Slice) Aggregate |
|---|---|---|
| **HBM2e Memory** | `32 GiB` @ `1,638 GB/s` | `512 GiB` @ `26.21 TB/s` |
| **VMEM (Vector SRAM)** | `128 MiB` (`131,072 KiB`) @ `10.24 TB/s` | `2.0 GiB` @ `163.84 TB/s` |
| **SMEM (Scalar SRAM)** | `1,024 KiB` (`1 MiB`) | `16 MiB` |
| **SparseCore SPMEM** | `512 KiB` (`2 × 256 KiB`) | `8 MiB` |
| **MXU (Systolic Matrix Engine)** | `2 × 256×256` @ `918 BF16 TFLOP/s` | `14.69 BF16 PFLOP/s` |
| **VPU & XLU (Vector / Cross-Lane)** | `128×8` VPU + `2×` XLU @ `28.7 FP32 TFLOP/s` | `459.2 FP32 TFLOP/s` |
| **2D Torus ICI Interconnect** | `4` links @ `720 GB/s` bidirectional | `11.52 TB/s` bidirectional |

---

## Overview of the 9 Benchmarks & Scientific / HPC / AI Use Cases

### 1. AI4Sim: Llama 3.1 405B & ScaFFold Surrogate (`bench_ai4sim`)
- **What Was Benchmarked**: Distributed Transformer block (`8` heads/chip, `seq_len = 4096`, `head_dim = 128`, `hidden_tp = 2048`, `ffn_tp = 8192`) sharded across the 2D `(dp, tp)` Torus mesh. Executes Pallas Multi-Head FlashAttention with query tiles pinned in VMEM (`q_scratch_ref`), MXU output projection, `SwiGLU` FFN, and 2D Tensor-Parallel `lax.psum` All-Reduce over ICI.
- **Use Case**: Large-scale LLM training/inference (MLPerf Llama 3.1 405B) and LLNL's **ScaFFold** neural physics operators that steer or accelerate multi-physics simulations in inertial confinement fusion (ICF) and materials design.

### 2. ModSim: Laghos / Remhos / MiniEM 3D High-Order FEM (`bench_modsim_fem`)
- **What Was Benchmarked**: Matrix-free high-order 3D Finite Element Method (FEM) sum-factorization across `8.39M` DOFs/chip (`134.2M` DOFs across 16 chips) using `1024×256` MXU tensor contractions overlapped with boundary slab `lax.ppermute` halo exchanges over ICI.
- **Use Case**: Compressible shock hydrodynamics (**Laghos**), Arbitrary Lagrangian-Eulerian (ALE) mesh remapping (**Remhos**), and pulsed-power/plasma electromagnetics (**MiniEM**).

### 3. ModSim: LAMMPS (ACE) Molecular Dynamics (`bench_modsim_lammps`)
- **What Was Benchmarked**: Atomic Cluster Expansion (ACE) interatomic potential force evaluation over `65,536` atoms × `256` neighbor descriptors per chip (`1.05M` atoms across 16 chips), combining VPU radial basis evaluation with MXU multi-body tensor contractions in VMEM.
- **Use Case**: Million-atom ab-initio-accuracy molecular dynamics for phase transitions, shock compression, and radiation damage in metals and actinides under extreme conditions.

### 4. ModSim: RAJA Performance Suite — Hydrodynamics (`bench_modsim_raja`)
- **What Was Benchmarked**: Memory-bandwidth-bound predictor-corrector hydrodynamics energy/pressure update loop over `67.1M` zones per chip (`1.07 billion` zones across 16 chips) streaming `1.024 GB`/chip through double-buffered `512×1024` (`2 MiB`) VMEM tiles.
- **Use Case**: Evaluates sustained HBM2e streaming bandwidth, async DMA double-buffering efficiency, and VPU vectorization on low-arithmetic-intensity multiphysics stencil/loop kernels (**LULESH / Ares / Kull**).

### 5. ModSim: Kripke 3D $S_n$ Deterministic Particle Transport (`bench_modsim_kripke`)
- **What Was Benchmarked**: 3D discrete-ordinates ($S_n$) diamond-difference wavefront sweep across `64` spatial zone slabs × `512` energy groups × `256` angles (`8.39M` unknowns/chip, `134.2M` unknowns across 16 chips) with upwind angular fluxes (`scratch_psi_ref`) pinned in VMEM.
- **Use Case**: Deterministic neutron and photon radiation transport solving the linear Boltzmann equation for nuclear energy, shielding, and fusion target energetics.

### 6. ModSim: Branson (Monte Carlo) & SPARTA (DSMC) (`bench_modsim_branson_sparta`)
- **What Was Benchmarked**: Branchless vectorized particle push and collision distance tracking across `33.55M` particles/chip (`536.8M` particles across 16 chips) overlapped with cross-chip particle migration (`lax.ppermute`) over ICI.
- **Use Case**: Implicit Monte Carlo thermal photon transport in high-energy-density plasmas (**Branson**) and Direct Simulation Monte Carlo rarefied gas dynamics for hypersonic reentry (**SPARTA**).

### 7. ModSim: AMG2023 Algebraic Multigrid Solver (`bench_modsim_amg2023`)
- **What Was Benchmarked**: Fused 3-step Chebyshev polynomial smoother + Block-CSR MXU SpMV over `8.39M` unknowns/chip (`134.2M` unknowns across 16 chips), executing 3 smoothing iterations per VMEM residency pass to cut HBM traffic by $3\times$.
- **Use Case**: Scalable implicit linear solvers ($Ax = b$) for unstructured radiation diffusion, incompressible flow, and structural mechanics.

### 8. Collectives: MPI — OMB / Phloem / Sandia ISx (`bench_collectives_omb_isx`)
- **What Was Benchmarked**: 16-chip global `lax.all_to_all` bucket exchange across the `4x4` 2D Torus ICI interconnect followed by vectorized local radix/bitonic sorting (`lax.sort`) over `16.78M` 32-bit integer keys/chip (`268.4M` keys / `1.07 GB` globally).
- **Use Case**: Multi-host ICI bisection bandwidth (**OMB / Phloem**) and distributed key redistribution/sorting (**Sandia ISx**) for particle re-binning and Adaptive Mesh Refinement (AMR) load balancing.

### 9. Storage: IO500 & MLPerf Storage Checkpoint Stream (`bench_storage_io500`)
- **What Was Benchmarked**: Streaming DMA read/transform/writeback pipeline over `256 MiB` input + `256 MiB` output per chip (`8.0 GiB` per step across 16 chips) using `1024×1024` (`4 MiB`) double-buffered VMEM tiles.
- **Use Case**: High-frequency defensive simulation checkpoint/restart (**IO500**) and high-bandwidth AI training data ingestion (**MLPerf Storage / DLIO**).

---

## 16-Chip (`4x4`) Cloud TPU v6e Benchmark Results

### Table 1: Compute Units (`MXU` & `VPU/XLU`) & Overall Figure of Merit (FOM)

| Workload | Step Time (ms) | Compile (ms) | Primary Figure of Merit (FOM) | MXU (TFLOP/s/chip) | MXU Util (%) | VPU/XLU (TFLOP/s/chip) | VPU Util (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | **1.64** | 15.5 | **4,888.13 TFLOP/s** | 304.6 | 33.2% | 0.92 | 3.2% |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | **0.40** | 2.9 | **333.69 GDOFs/s** | 32.0 | 3.5% | 0.04 | 0.1% |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | **0.42** | 2.7 | **2,497.68 MAtom-Steps/s** | 40.9 | 4.5% | 0.40 | 1.4% |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | **1.04** | 2.2 | **16,462.91 GB/s Stream BW** | 0.0 | 0.0% | 0.58 | 2.0% |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | **0.41** | 1.6 | **326.29 GUnknowns/s** | 0.0 | 0.0% | 0.14 | 0.5% |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | **0.69** | 6.5 | **779.29 GParticles/s** | 0.0 | 0.0% | 0.29 | 1.0% |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | **0.40** | 1.8 | **15,925.18 GB/s Effective 3-Step SpMV BW** | 31.9 | 3.5% | 0.37 | 1.3% |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | **19.44** | 126.5 | **55.23 GB/s Global Sort+Exch** | 0.0 | 0.0% | 0.18 | 0.6% |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | **0.70** | 2.2 | **12,188.63 GB/s Aggregate DMA BW** | 0.0 | 0.0% | 0.19 | 0.7% |

### Table 2: Memory Hierarchy (`HBM`, `VMEM`, `SMEM`, `SPMEM`) Usage & Bandwidth

| Workload | HBM Base / Peak (MB) | HBM Overhead | HBM BW / Chip (GB/s) | HBM Util (%) | VMEM Used (KiB / %) | VMEM BW (TB/s) | SMEM (KiB) | SparseCore SPMEM (KiB / %) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | 140 / 285 | **2.03x** | 103.2 | 6.3% | 32768 (25.0%) | 0.29 | 64 | 0 (0.0%) |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | 48 / 113 | **2.36x** | 208.6 | 12.7% | 12288 (9.4%) | 0.67 | 32 | 0 (0.0%) |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | 64 / 193 | **3.02x** | 319.7 | 19.5% | 10240 (7.8%) | 0.96 | 32 | 128 (25.0%) |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | 768 / 1281 | **1.67x** | 1028.9 | 62.8% | 16384 (12.5%) | 2.06 | 16 | 0 (0.0%) |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | 64 / 128 | **2.01x** | 244.7 | 14.9% | 4096 (3.1%) | 0.39 | 24 | 0 (0.0%) |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | 384 / 641 | **1.67x** | 779.3 | 47.6% | 16384 (12.5%) | 1.56 | 48 | 384 (75.0%) |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | 96 / 161 | **1.68x** | 331.8 | 20.3% | 12288 (9.4%) | 1.00 | 32 | 256 (50.0%) |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | 64 / 197 | **3.09x** | 6.9 | 0.4% | 65536 (50.0%) | 0.07 | 128 | 448 (87.5%) |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | 256 / 770 | **3.01x** | 761.8 | 46.5% | 16384 (12.5%) | 0.76 | 16 | 0 (0.0%) |

### Table 3: Data Movement Engines (`DMA` & `2D Torus ICI`) & `.xplane.pb` Trace Files

| Workload | DMA Volume / Chip (MB) | DMA Throughput / Chip (GB/s) | Compute/DMA Overlap (%) | ICI Volume / Chip (MB) | ICI BW / Chip (GB/s) | 16-Chip ICI BW (GB/s) | ICI Util (%) | `.xplane.pb` Files |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | 185.0 | 118.6 | **95.2%** | 24.0 | 15.4 | **246.2** | 2.1% | 1 (`ai4sim_llama_scaffold/`) |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | 80.5 | 209.9 | **96.2%** | 0.5 | 1.3 | **20.9** | 0.2% | 1 (`modsim_fem_laghos/`) |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | 128.0 | 319.7 | **98.0%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`modsim_lammps_ace/`) |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | 1024.0 | 1028.9 | **99.2%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`modsim_raja_hydro/`) |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | 97.0 | 247.3 | **98.9%** | 1.0 | 2.5 | **40.8** | 0.4% | 1 (`modsim_kripke_sn/`) |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | 516.0 | 785.4 | **99.2%** | 4.0 | 6.1 | **97.4** | 0.8% | 1 (`modsim_branson_sparta/`) |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | 128.5 | 333.1 | **97.6%** | 0.5 | 1.3 | **20.7** | 0.2% | 1 (`modsim_amg2023/`) |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | 248.0 | 13.4 | **91.6%** | 120.0 | 6.5 | **103.6** | 0.9% | 1 (`collectives_omb_isx/`) |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | 512.0 | 761.8 | **99.2%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`storage_io500_mlperf/`) |

---

## Running on GKE (`4x4` Cloud TPU v6e Node Pool)

```bash
# 1. Package benchmark scripts into the GKE ConfigMap
kubectl create configmap ats6-tpuv6e-code \
  --from-file=ats6_pallas_benchmarks.py \
  --from-file=ats6_tpuc_benchmarks.py \
  --from-file=ats6_tpuc_kernels.cc \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Launch the 4-worker (16-chip) Indexed Job and Headless Service
kubectl delete job llnl-ats6-tpuv6e-benchmark --ignore-not-found
kubectl apply -f gke_tpu_v6e_job.yaml

# 3. Wait for completion and inspect results
kubectl wait --for=condition=complete job/llnl-ats6-tpuv6e-benchmark --timeout=300s
for p in $(kubectl get pods -l app=llnl-ats6-tpuv6e-benchmark -o name); do
  kubectl logs "$p"
done
```
