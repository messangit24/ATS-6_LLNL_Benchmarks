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
  - A distributed Transformer block (`seq_len = 4096`, `head_dim_packed = 256`, `hidden_tp = 2048`, `ffn_tp = 8192`) sharded across the 2D `(dp, tp)` Torus mesh.
  - Custom Pallas Multi-Head FlashAttention (`_flash_attention_v6e_kernel`) that packs head pairs into `1024×256` query blocks and FP32 output accumulators pinned in on-chip VMEM (`q_scratch_ref`, `o_scratch_ref`) across `512×256` KV tiles to fill 100% of TPU v6e's `256×256` systolic MXUs, followed by a zero-copy attention output projection whose Tensor-Parallel `lax.psum` All-Reduce over ICI is overlapped concurrently with the `SwiGLU` feed-forward network on the MXUs.
  - **Achieved Performance**: **`7,798.09 TFLOP/s` (`7.80 PFLOP/s`)** (`1.02 ms` step latency, `486.6 TFLOP/s/chip` = **53.0% MXU**, `393.2 GB/s` 16-chip ICI bandwidth).
- **Use Case**:
  - **AI-for-Science Foundation Models & Neural Physics Surrogates**: Large-scale LLM training/inference (MLPerf Llama 3.1 405B) and LLNL's **ScaFFold** spatial/temporal neural operators that steer or replace expensive multi-physics simulations in inertial confinement fusion (ICF), materials discovery, and hydrodynamic design.

### 2. ModSim: Laghos / Remhos / MiniEM 3D High-Order FEM (`bench_modsim_fem`)
- **What Was Benchmarked**:
  - Matrix-free high-order 3D Finite Element Method (FEM) 2-stage RK2 sum-factorization (`_fem_v6e_kernel`) across `33.55M` degrees of freedom (DOFs) per chip (`536.9M` DOFs across 16 chips).
  - Streams state and geometric factors in BF16 from HBM and casts 6-stage 3D element gradient/divergence tensor contractions (`4096×256` super-tiles) with FP32 accumulation onto TPU v6e's `256×256` systolic MXUs inside VMEM while overlapping boundary slab `lax.ppermute` halo exchanges over ICI without intermediate HBM round-trips.
  - **Achieved Performance**: **`5,289.11 GDOFs/s` (`5.29 TDOFs/s`)** (`0.20 ms` step latency, `507.8 TFLOP/s/chip` = **55.3% MXU**, `991.7 GB/s/chip` = **60.5% HBM**, `4.63 TB/s/chip` VMEM bandwidth).
- **Use Case**:
  - **Compressible Shock Hydrodynamics & Electromagnetics**:
    - **Laghos** (Lagrangian high-order solver): Models shock-wave propagation, implosions, and high-deformation fluid flow on moving unstructured meshes.
    - **Remhos** (Remap high-order solver): Advection-based Arbitrary Lagrangian-Eulerian (ALE) mesh remapping.
    - **MiniEM**: Maxwell's equations solver for pulsed-power and plasma electromagnetics.

### 3. ModSim: LAMMPS (ACE) Molecular Dynamics (`bench_modsim_lammps`)
- **What Was Benchmarked**:
  - Atomic Cluster Expansion (ACE) machine-learned interatomic potential force evaluation (`_lammps_ace_v6e_kernel`) over `131,072` atoms × `256` neighbor descriptors per chip (`2.10M` atoms across 16 chips) using `4096×256` VMEM super-tiles.
  - Streams neighbor descriptors in BF16, upcasts in VMEM to evaluate radial basis functions (`exp2(-r) * (1 + r + 0.5 r^2)`) on the FP32 Vector Processing Unit (VPU), and contracts 4-stage multi-body basis weights and adjoint forces with FP32 accumulation on the Matrix Multiply Unit (MXU) inside VMEM.
  - **Achieved Performance**: **`13,720.91 MAtom-Steps/s` (`13.72 GAtom-Steps/s`)** (`0.15 ms` step latency, `449.6 TFLOP/s/chip` = **49.0% MXU**, `878.1 GB/s/chip` = **53.6% HBM**, `4.39 TB/s/chip` VMEM bandwidth).
- **Use Case**:
  - **Ab-Initio-Accuracy Materials Science at Million-Atom Scale**: Simulating phase transitions, shock compression, radiation damage, and defect migration in metals, actinides, and carbon under extreme pressures and temperatures.

### 4. ModSim: RAJA Performance Suite — Hydrodynamics (`bench_modsim_raja`)
- **What Was Benchmarked**:
  - Memory-bandwidth-bound predictor-corrector hydrodynamics energy/pressure update loop (`_raja_hydro_v6e_kernel`) over `67.1M` zones per chip (`1.07 billion` zones across 16 chips).
  - Streams three FP32 input fields (`e`, `p`, `dv`) and one FP32 output field (`1.024 GB` DMA traffic per chip) through double-buffered `1024×1024` (`4 MiB`) VMEM super-tiles (`32 MiB` active VMEM).
  - **Achieved Performance**: **`17,566.07 GB/s` (`17.57 TB/s`) Stream BW** across 16 chips (`0.98 ms` step latency, `1,097.9 GB/s/chip` = **67.0%** of raw HBM2e peak bandwidth).
- **Use Case**:
  - **Portable HPC Stencil & Loop Kernels (LULESH / Ares / Kull)**: Evaluates sustained HBM2e streaming bandwidth, async DMA double-buffering efficiency, and VPU vectorization on structured/unstructured multiphysics loops where arithmetic intensity is low (`<1 FLOP/byte`).

### 5. ModSim: Kripke 3D $S_n$ Deterministic Particle Transport (`bench_modsim_kripke`)
- **What Was Benchmarked**:
  - 3D discrete-ordinates ($S_n$) diamond-difference wavefront sweep (`_kripke_sweep_v6e_kernel`) across `32` spatial zone slabs × `1024` energy groups × `512` angular directions (`16.78M` unknowns/chip, `268.4M` unknowns across 16 chips).
  - Uses `dimension_semantics=("arbitrary",)` to enforce sequential wavefront dependencies across `2 MiB` zone slabs while pinning the upwind angular flux (`scratch_psi_ref`) in on-chip VMEM and overlapping ICI boundary flux exchange.
  - **Achieved Performance**: **`1,397.61 GUnknowns/s` (`1.40 TUnknowns/s`)** (`0.19 ms` step latency, `1,048.2 GB/s/chip` = **64.0% HBM**, `349.4 GB/s` 16-chip ICI bandwidth).
- **Use Case**:
  - **Deterministic Neutron & Photon Radiation Transport**: Solves the linear Boltzmann transport equation for nuclear reactor physics, radiation shielding, and inertial confinement fusion target energetics.

### 6. ModSim: Branson (Monte Carlo) & SPARTA (DSMC) (`bench_modsim_branson_sparta`)
- **What Was Benchmarked**:
  - Stochastic particle push and distance-to-collision tracking (`_branson_sparta_v6e_kernel`) across `33.55M` particles per chip (`536.8M` particles across 16 chips) using `1024×1024` (`4 MiB`) VMEM super-tiles.
  - Replaces divergent per-particle branch conditionals (`if step < d_coll`) with branchless SIMD `jnp.minimum(step, d_coll)` across 1,024-wide VPU lanes while concurrently overlapping cross-chip particle migration (`lax.ppermute`) over ICI.
  - **Achieved Performance**: **`1,148.44 GParticles/s` (`1.15 TParticles/s`)** (`0.47 ms` step latency, `1,148.4 GB/s/chip` = **70.1% HBM**, `287.1 GB/s` 16-chip ICI bandwidth).
- **Use Case**:
  - **Stochastic Thermal Radiation & Rarefied Gas Dynamics**:
    - **Branson**: Implicit Monte Carlo (IMC) photon transport in high-energy-density plasmas.
    - **SPARTA** (Direct Simulation Monte Carlo): Hypersonic reentry, upper-atmosphere rarefied gas dynamics, and plasma-surface interactions.

### 7. ModSim: AMG2023 Algebraic Multigrid Solver (`bench_modsim_amg2023`)
- **What Was Benchmarked**:
  - Fused 3-step Chebyshev polynomial smoother + Block-CSR Sparse Matrix-Vector Multiply (SpMV) (`_amg2023_v6e_kernel`) over `33.55M` unknowns per chip (`536.9M` unknowns across 16 chips) using `4096×256` (`2 MiB`) BF16-streamed VMEM super-tiles.
  - Reformulates block-sparse coupling operators into `256×256` dense blocks on the MXU and fuses 3 FP32 Chebyshev smoothing iterations ($x^{(1)}, x^{(2)}, x^{(3)}$) inside a single VMEM residency pass to cut HBM traffic by $6\times$.
  - **Achieved Performance**: **`102,949.50 GB/s` (`102.95 TB/s`) Effective 3-Step SpMV BW** (`0.25 ms` step latency, `1,072.4 GB/s/chip` = **65.5% physical HBM**, `205.9 TFLOP/s/chip` = **22.4% MXU**).
- **Use Case**:
  - **Implicit Linear Solvers for Unstructured Multi-Physics**: Solves large sparse linear systems ($Ax = b$) arising from implicit radiation diffusion, incompressible fluid flow, and structural mechanics at extreme scale.

### 8. Collectives: MPI — OMB / Phloem / Sandia ISx (`bench_collectives_omb_isx`)
- **What Was Benchmarked**:
  - A 16-chip global `lax.all_to_all` bucket exchange across the `4x4` 2D Torus ICI interconnect followed by a custom Pallas VMEM `4096×128` (`2 MiB` super-tile) vectorized bucket/order-statistic pass (`_isx_vmem_bucket_kernel`) on `67.11M` 32-bit integer keys per chip (`1.07 billion` keys / `4.29 GB` globally).
  - **Achieved Performance**: **`1,289.94 GB/s` (`1.29 TB/s`) Global Sort+Exch** (`3.33 ms` step latency, **`2,418.6 GB/s` (`2.42 TB/s`) 16-chip ICI bandwidth** = `151.2 GB/s/chip`).
- **Use Case**:
  - **Bisection Bandwidth & Distributed Data Redistribution**:
    - **OSU Micro-Benchmarks (OMB) & Phloem**: Measures multi-host ICI bisection bandwidth and all-to-all collective congestion.
    - **Sandia ISx (Integer Sort)**: Models global particle re-binning, adaptive mesh refinement (AMR) load balancing, and distributed graph partitioning.

### 9. Storage: IO500 & MLPerf Storage Checkpoint Stream (`bench_storage_io500`)
- **What Was Benchmarked**:
  - High-throughput streaming DMA read/transform/writeback pipeline (`_storage_v6e_kernel`) over `256 MiB` input + `256 MiB` output per chip (`8.0 GiB` per step across 16 chips) using `2048×1024` (`8 MiB`) double-buffered VMEM super-tiles (`32 MiB` active VMEM).
  - **Achieved Performance**: **`18,804.28 GB/s` (`18.80 TB/s`) Aggregate DMA BW** (`0.46 ms` step latency, `1,175.3 GB/s/chip` = **71.8%** of raw HBM2e peak bandwidth).
- **Use Case**:
  - **Defensive Simulation Checkpointing & AI Training Data Ingest**:
    - **IO500**: High-frequency defensive state checkpoint/restart for long-running hydro/transport runs.
    - **MLPerf Storage (DLIO)**: Sustaining multi-gigabyte/s training sample ingestion and model weight checkpointing without stalling TPU compute engines.

---

## Part 2: Complete 16-Chip Cloud TPU v6e Hardware Telemetry Tables

### Table 1: Compute Units (`MXU` & `VPU/XLU`) & Overall Figure of Merit (FOM)

| Workload | Step Time (ms) | Compile (ms) | Primary Figure of Merit (FOM) | MXU (TFLOP/s/chip) | MXU Util (%) | VPU/XLU (TFLOP/s/chip) | VPU Util (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | **1.02** | 73.6 | **7,798.09 TFLOP/s** | 486.6 | 53.0% | 0.82 | 2.9% |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | **0.20** | 2.1 | **5,289.11 GDOFs/s** | 507.8 | 55.3% | 0.66 | 2.3% |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | **0.15** | 12.0 | **13,720.91 MAtom-Steps/s** | 449.6 | 49.0% | 3.07 | 10.7% |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | **0.98** | 8.8 | **17,566.07 GB/s Stream BW** | 0.0 | 0.0% | 0.62 | 2.2% |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | **0.19** | 5.1 | **1,397.61 GUnknowns/s** | 0.0 | 0.0% | 0.61 | 2.1% |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | **0.47** | 5.2 | **1,148.44 GParticles/s** | 0.0 | 0.0% | 0.43 | 1.5% |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | **0.25** | 39.2 | **102,949.50 GB/s Effective 3-Step SpMV BW** | 205.9 | 22.4% | 2.41 | 8.4% |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | **3.33** | 5.6 | **1,289.94 GB/s Global Sort+Exch** | 0.0 | 0.0% | 0.08 | 0.3% |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | **0.46** | 2.8 | **18,804.28 GB/s Aggregate DMA BW** | 0.0 | 0.0% | 0.29 | 1.0% |

### Table 2: Memory Hierarchy (`HBM`, `VMEM`, `SMEM`, `SPMEM`) Usage & Bandwidth

| Workload | HBM Base / Peak (MB) | HBM Overhead | HBM BW / Chip (GB/s) | HBM Util (%) | VMEM Used (KiB / %) | VMEM BW (TB/s) | SMEM (KiB) | SparseCore SPMEM (KiB / %) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | 140 / 300 | **2.14x** | 164.9 | 10.1% | 32768 (25.0%) | 0.46 | 64 | 0 (0.0%) |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | 128 / 259 | **2.02x** | 991.7 | 60.5% | 24576 (18.8%) | 4.63 | 32 | 0 (0.0%) |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | 64 / 193 | **3.02x** | 878.1 | 53.6% | 20480 (15.6%) | 4.39 | 32 | 128 (25.0%) |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | 768 / 1282 | **1.67x** | 1097.9 | 67.0% | 32768 (25.0%) | 2.20 | 16 | 0 (0.0%) |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | 128 / 261 | **2.04x** | 1048.2 | 64.0% | 14336 (10.9%) | 1.68 | 24 | 0 (0.0%) |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | 384 / 649 | **1.69x** | 1148.4 | 70.1% | 32768 (25.0%) | 2.30 | 48 | 384 (75.0%) |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | 192 / 322 | **1.68x** | 1072.4 | 65.5% | 24576 (18.8%) | 3.22 | 32 | 256 (50.0%) |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | 256 / 770 | **3.01x** | 161.2 | 9.8% | 16384 (12.5%) | 0.16 | 128 | 448 (87.5%) |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | 256 / 770 | **3.01x** | 1175.3 | 71.8% | 32768 (25.0%) | 1.18 | 16 | 0 (0.0%) |

### Table 3: Data Movement Engines (`DMA` & `2D Torus ICI`) & `.xplane.pb` Trace Files

| Workload | DMA Volume / Chip (MB) | DMA Throughput / Chip (GB/s) | Compute/DMA Overlap (%) | ICI Volume / Chip (MB) | ICI BW / Chip (GB/s) | 16-Chip ICI BW (GB/s) | ICI Util (%) | `.xplane.pb` Files |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | 185.0 | 189.4 | **95.2%** | 24.0 | 24.6 | **393.2** | 3.4% | 1 (`ai4sim_llama_scaffold/`) |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | 193.0 | 996.9 | **96.9%** | 1.0 | 5.2 | **82.6** | 0.7% | 1 (`modsim_fem_laghos/`) |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | 128.0 | 878.1 | **96.1%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`modsim_lammps_ace/`) |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | 1024.0 | 1097.9 | **99.2%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`modsim_raja_hydro/`) |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | 196.0 | 1070.0 | **98.1%** | 4.0 | 21.8 | **349.4** | 3.0% | 1 (`modsim_kripke_sn/`) |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | 520.0 | 1166.4 | **98.4%** | 8.0 | 17.9 | **287.1** | 2.5% | 1 (`modsim_branson_sparta/`) |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | 256.5 | 1074.5 | **97.7%** | 0.5 | 2.1 | **33.5** | 0.3% | 1 (`modsim_amg2023/`) |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | 992.0 | 312.4 | **97.7%** | 480.0 | 151.2 | **2418.6** | 21.0% | 1 (`collectives_omb_isx/`) |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | 512.0 | 1175.3 | **98.4%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`storage_io500_mlperf/`) |
