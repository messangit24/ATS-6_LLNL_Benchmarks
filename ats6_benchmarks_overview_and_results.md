# LLNL ATS-6 Cloud TPU v6e (`4x4` = 16 Chips) Benchmark Overview, Use Cases & Performance Telemetry

- **Target GKE Cluster**: `mb-tpu-benchmark` (`mb-tpu-v6e-multi-16`)
- **Cluster Topology**: `16` Cloud TPU v6e (Trillium) chips (`4` hosts × `4` chips/host, `4x4` 2D Torus)
- **Execution Mode**: Unified 16-Chip SPMD Program (`shard_map` + Pallas VMEM Kernels + 2D Torus ICI Collectives)
- **Benchmark Source**: [`ats6_pallas_benchmarks.py`](./ats6_pallas_benchmarks.py)
- **Raw Telemetry JSON**: [`ats6_xprof_summary.json`](./ats6_xprof_summary.json)

---

## Part 1: What Was Benchmarked & Scientific / HPC / AI Use Cases

The 9 benchmarks represent the **LLNL ATS-6 (Advanced Technology System 6)** procurement workload categories spanning **AI-for-Science (`AI4Sim`)**, **Modeling & Simulation (`ModSim`)**, **Interconnect Collectives**, and **Storage/Checkpointing**.

### 1. AI4Sim: Llama 3.1 405B & ScaFFold Surrogate (`bench_ai4sim`)
- **What Was Benchmarked**:
  - A distributed Transformer block (`8` heads/chip, `seq_len = 4096`, `head_dim = 128`, `hidden_tp = 2048`, `ffn_tp = 8192`) sharded across the 2D `(dp, tp)` Torus mesh.
  - Custom Pallas Multi-Head FlashAttention (`_flash_attention_v6e_kernel`) that pins `512x128` query blocks in on-chip VMEM (`q_scratch_ref`) across `512x128` KV tiles, followed by an attention output projection, a `SwiGLU` feed-forward network on the dual `256x256` MXUs, and a 2D Tensor-Parallel `lax.psum` All-Reduce over ICI.
  - **Achieved Performance**: **`4,888.13 TFLOP/s`** (`1.64 ms` step latency, `304.6 TFLOP/s/chip` MXU, `246.2 GB/s` 16-chip ICI bandwidth).
- **Use Case**:
  - **AI-for-Science Foundation Models & Neural Physics Surrogates**: Large-scale LLM training/inference (MLPerf Llama 3.1 405B) and LLNL's **ScaFFold** spatial/temporal neural operators that steer or replace expensive multi-physics simulations in inertial confinement fusion (ICF), materials discovery, and hydrodynamic design.

### 2. ModSim: Laghos / Remhos / MiniEM 3D High-Order FEM (`bench_modsim_fem`)
- **What Was Benchmarked**:
  - Matrix-free high-order 3D Finite Element Method (FEM) sum-factorization (`_fem_v6e_kernel`) across `8.39M` degrees of freedom (DOFs) per chip (`134.2M` DOFs across 16 chips).
  - Casts 3D element gradient/divergence tensor contractions (`u @ D`, `du_dx @ D^T`, `q_flux @ D`) onto TPU v6e's `256x256` systolic MXUs (`1024x256` super-tiles) while overlapping boundary slab `lax.ppermute` halo exchanges over ICI.
  - **Achieved Performance**: **`333.69 GDOFs/s`** (`0.40 ms` step latency, `208.6 GB/s/chip` HBM bandwidth, `20.9 GB/s` 16-chip ICI bandwidth).
- **Use Case**:
  - **Compressible Shock Hydrodynamics & Electromagnetics**:
    - **Laghos** (Lagrangian high-order solver): Models shock-wave propagation, implosions, and high-deformation fluid flow on moving unstructured meshes.
    - **Remhos** (Remap high-order solver): Advection-based Arbitrary Lagrangian-Eulerian (ALE) mesh remapping.
    - **MiniEM**: Maxwell's equations solver for pulsed-power and plasma electromagnetics.

### 3. ModSim: LAMMPS (ACE) Molecular Dynamics (`bench_modsim_lammps`)
- **What Was Benchmarked**:
  - Atomic Cluster Expansion (ACE) machine-learned interatomic potential force evaluation (`_lammps_ace_v6e_kernel`) over `65,536` atoms × `256` neighbor descriptors per chip (`1.05M` atoms across 16 chips).
  - Evaluates radial basis functions (`exp2(-r) * (1 + r + 0.5 r^2)`) on the FP32 Vector Processing Unit (VPU) and contracts multi-body basis weights (`B @ W` and `C @ W^T`) on the BF16 Matrix Multiply Unit (MXU) inside VMEM.
  - **Achieved Performance**: **`2,497.68 MAtom-Steps/s`** (`0.42 ms` step latency, `40.9 TFLOP/s/chip` MXU, `319.7 GB/s/chip` HBM bandwidth).
- **Use Case**:
  - **Ab-Initio-Accuracy Materials Science at Million-Atom Scale**: Simulating phase transitions, shock compression, radiation damage, and defect migration in metals, actinides, and carbon under extreme pressures and temperatures.

### 4. ModSim: RAJA Performance Suite — Hydrodynamics (`bench_modsim_raja`)
- **What Was Benchmarked**:
  - Memory-bandwidth-bound predictor-corrector hydrodynamics energy/pressure update loop (`_raja_hydro_v6e_kernel`) over `67.1M` zones per chip (`1.07 billion` zones across 16 chips).
  - Streams three FP32 input fields (`e`, `p`, `dv`) and one FP32 output field (`1.024 GB` DMA traffic per chip) through double-buffered `512x1024` (`2 MiB`) VMEM tiles.
  - **Achieved Performance**: **`16,462.91 GB/s` Stream BW** across 16 chips (`1.04 ms` step latency, `1,028.9 GB/s/chip` = **62.8%** of raw HBM2e peak bandwidth).
- **Use Case**:
  - **Portable HPC Stencil & Loop Kernels (LULESH / Ares / Kull)**: Evaluates sustained HBM2e streaming bandwidth, async DMA double-buffering efficiency, and VPU vectorization on structured/unstructured multiphysics loops where arithmetic intensity is low (`<1 FLOP/byte`).

### 5. ModSim: Kripke 3D $S_n$ Deterministic Particle Transport (`bench_modsim_kripke`)
- **What Was Benchmarked**:
  - 3D discrete-ordinates ($S_n$) diamond-difference wavefront sweep (`_kripke_sweep_v6e_kernel`) across `64` spatial zone slabs × `512` energy groups × `256` angular directions (`8.39M` unknowns/chip, `134.2M` unknowns across 16 chips).
  - Uses `dimension_semantics=("arbitrary",)` to enforce sequential wavefront dependencies across zone slabs while pinning the upwind angular flux (`scratch_psi_ref`) in on-chip VMEM so intermediate wavefront fluxes never round-trip to HBM.
  - **Achieved Performance**: **`326.29 GUnknowns/s`** (`0.41 ms` step latency, `244.7 GB/s/chip` HBM bandwidth, `40.8 GB/s` 16-chip ICI bandwidth).
- **Use Case**:
  - **Deterministic Neutron & Photon Radiation Transport**: Solves the linear Boltzmann transport equation for nuclear reactor physics, radiation shielding, and inertial confinement fusion target energetics.

### 6. ModSim: Branson (Monte Carlo) & SPARTA (DSMC) (`bench_modsim_branson_sparta`)
- **What Was Benchmarked**:
  - Stochastic particle push and distance-to-collision tracking (`_branson_sparta_v6e_kernel`) across `33.55M` particles per chip (`536.8M` particles across 16 chips).
  - Replaces divergent per-particle branch conditionals (`if step < d_coll`) with branchless SIMD `jnp.minimum(step, d_coll)` across 1,024-wide VPU lanes while overlapping cross-chip particle migration (`lax.ppermute`) over ICI.
  - **Achieved Performance**: **`779.29 GParticles/s`** (`0.69 ms` step latency, `779.3 GB/s/chip` HBM bandwidth, `97.4 GB/s` 16-chip ICI bandwidth).
- **Use Case**:
  - **Stochastic Thermal Radiation & Rarefied Gas Dynamics**:
    - **Branson**: Implicit Monte Carlo (IMC) photon transport in high-energy-density plasmas.
    - **SPARTA** (Direct Simulation Monte Carlo): Hypersonic reentry, upper-atmosphere rarefied gas dynamics, and plasma-surface interactions.

### 7. ModSim: AMG2023 Algebraic Multigrid Solver (`bench_modsim_amg2023`)
- **What Was Benchmarked**:
  - Fused 3-step Chebyshev polynomial smoother + Block-CSR Sparse Matrix-Vector Multiply (SpMV) (`_amg2023_v6e_kernel`) over `8.39M` unknowns per chip (`134.2M` unknowns across 16 chips).
  - Reformulates block-sparse coupling operators into `256x256` dense blocks on the MXU and fuses 3 Chebyshev smoothing iterations ($x^{(1)}, x^{(2)}, x^{(3)}$) inside a single VMEM residency pass to cut HBM traffic by $3\times$.
  - **Achieved Performance**: **`15,925.18 GB/s` Effective 3-Step SpMV BW** (`0.40 ms` step latency, `331.8 GB/s/chip` physical HBM bandwidth, `31.9 TFLOP/s/chip` MXU).
- **Use Case**:
  - **Implicit Linear Solvers for Unstructured Multi-Physics**: Solves large sparse linear systems ($Ax = b$) arising from implicit radiation diffusion, incompressible fluid flow, and structural mechanics at extreme scale.

### 8. Collectives: MPI — OMB / Phloem / Sandia ISx (`bench_collectives_omb_isx`)
- **What Was Benchmarked**:
  - A 16-chip global `lax.all_to_all` bucket exchange across the `4x4` 2D Torus ICI interconnect followed by a local vectorized radix/bitonic sort (`lax.sort`) over `16.78M` 32-bit integer keys per chip (`268.4M` keys / `1.07 GB` globally).
  - **Achieved Performance**: **`55.23 GB/s` Global Sort+Exch** (`19.44 ms` step latency, `103.6 GB/s` 16-chip ICI bandwidth).
- **Use Case**:
  - **Bisection Bandwidth & Distributed Data Redistribution**:
    - **OSU Micro-Benchmarks (OMB) & Phloem**: Measures multi-host ICI bisection bandwidth and all-to-all collective congestion.
    - **Sandia ISx (Integer Sort)**: Models global particle re-binning, adaptive mesh refinement (AMR) load balancing, and distributed graph partitioning.

### 9. Storage: IO500 & MLPerf Storage Checkpoint Stream (`bench_storage_io500`)
- **What Was Benchmarked**:
  - High-throughput streaming DMA read/transform/writeback pipeline (`_storage_v6e_kernel`) over `256 MiB` input + `256 MiB` output per chip (`8.0 GiB` per step across 16 chips) using `1024x1024` (`4 MiB`) double-buffered VMEM tiles.
  - **Achieved Performance**: **`12,188.63 GB/s` Aggregate DMA BW** (`0.70 ms` step latency, `761.8 GB/s/chip` DMA/HBM throughput).
- **Use Case**:
  - **Defensive Simulation Checkpointing & AI Training Data Ingest**:
    - **IO500**: High-frequency defensive state checkpoint/restart for long-running hydro/transport runs.
    - **MLPerf Storage (DLIO)**: Sustaining multi-gigabyte/s training sample ingestion and model weight checkpointing without stalling TPU compute engines.

---

## Part 2: Complete 16-Chip Cloud TPU v6e Hardware Telemetry Tables

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
