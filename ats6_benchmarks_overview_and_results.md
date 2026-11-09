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
  - Custom Pallas Multi-Head FlashAttention (`_flash_attention_v6e_kernel`) that packs head pairs into `1024×256` query blocks pinned in on-chip VMEM (`q_scratch_ref`) across `512×256` KV tiles to fill 100% of TPU v6e's `256×256` systolic MXUs, followed by a zero-copy attention output projection, a fused `w_gate_up` `SwiGLU` feed-forward network, and a 2D Tensor-Parallel `lax.psum` All-Reduce over ICI.
  - **Achieved Performance**: **`7,183.79 TFLOP/s` (`7.18 PFLOP/s`)** (`1.11 ms` step latency, `448.2 TFLOP/s/chip` = **48.8% MXU**, `362.3 GB/s` 16-chip ICI bandwidth).
- **Use Case**:
  - **AI-for-Science Foundation Models & Neural Physics Surrogates**: Large-scale LLM training/inference (MLPerf Llama 3.1 405B) and LLNL's **ScaFFold** spatial/temporal neural operators that steer or replace expensive multi-physics simulations in inertial confinement fusion (ICF), materials discovery, and hydrodynamic design.

### 2. ModSim: Laghos / Remhos / MiniEM 3D High-Order FEM (`bench_modsim_fem`)
- **What Was Benchmarked**:
  - Matrix-free high-order 3D Finite Element Method (FEM) 2-stage RK2 sum-factorization (`_fem_v6e_kernel`) across `16.78M` degrees of freedom (DOFs) per chip (`268.4M` DOFs across 16 chips).
  - Casts 6-stage 3D element gradient/divergence tensor contractions (`2048×256` super-tiles) onto TPU v6e's `256×256` systolic MXUs inside VMEM while overlapping boundary slab `lax.ppermute` halo exchanges over ICI without intermediate HBM round-trips.
  - **Achieved Performance**: **`2,378.67 GDOFs/s`** (`0.23 ms` step latency, `228.4 TFLOP/s/chip` = **24.9% MXU**, `743.3 GB/s/chip` = **45.4% HBM**, `4.16 TB/s/chip` VMEM bandwidth).
- **Use Case**:
  - **Compressible Shock Hydrodynamics & Electromagnetics**:
    - **Laghos** (Lagrangian high-order solver): Models shock-wave propagation, implosions, and high-deformation fluid flow on moving unstructured meshes.
    - **Remhos** (Remap high-order solver): Advection-based Arbitrary Lagrangian-Eulerian (ALE) mesh remapping.
    - **MiniEM**: Maxwell's equations solver for pulsed-power and plasma electromagnetics.

### 3. ModSim: LAMMPS (ACE) Molecular Dynamics (`bench_modsim_lammps`)
- **What Was Benchmarked**:
  - Atomic Cluster Expansion (ACE) machine-learned interatomic potential force evaluation (`_lammps_ace_v6e_kernel`) over `65,536` atoms × `256` neighbor descriptors per chip (`1.05M` atoms across 16 chips) using `2048×256` VMEM super-tiles.
  - Evaluates radial basis functions (`exp2(-r) * (1 + r + 0.5 r^2)`) on the FP32 Vector Processing Unit (VPU) and contracts 4-stage multi-body basis weights and adjoint forces on the BF16 Matrix Multiply Unit (MXU) inside VMEM.
  - **Achieved Performance**: **`5,436.61 MAtom-Steps/s`** (`0.19 ms` step latency, `178.1 TFLOP/s/chip` = **19.4% MXU**, `695.9 GB/s/chip` = **42.5% HBM**, `3.48 TB/s/chip` VMEM bandwidth).
- **Use Case**:
  - **Ab-Initio-Accuracy Materials Science at Million-Atom Scale**: Simulating phase transitions, shock compression, radiation damage, and defect migration in metals, actinides, and carbon under extreme pressures and temperatures.

### 4. ModSim: RAJA Performance Suite — Hydrodynamics (`bench_modsim_raja`)
- **What Was Benchmarked**:
  - Memory-bandwidth-bound predictor-corrector hydrodynamics energy/pressure update loop (`_raja_hydro_v6e_kernel`) over `67.1M` zones per chip (`1.07 billion` zones across 16 chips).
  - Streams three FP32 input fields (`e`, `p`, `dv`) and one FP32 output field (`1.024 GB` DMA traffic per chip) through double-buffered `1024×1024` (`4 MiB`) VMEM super-tiles (`32 MiB` active VMEM).
  - **Achieved Performance**: **`18,690.29 GB/s` (`18.69 TB/s`) Stream BW** across 16 chips (`0.92 ms` step latency, `1,168.1 GB/s/chip` = **71.3%** of raw HBM2e peak bandwidth).
- **Use Case**:
  - **Portable HPC Stencil & Loop Kernels (LULESH / Ares / Kull)**: Evaluates sustained HBM2e streaming bandwidth, async DMA double-buffering efficiency, and VPU vectorization on structured/unstructured multiphysics loops where arithmetic intensity is low (`<1 FLOP/byte`).

### 5. ModSim: Kripke 3D $S_n$ Deterministic Particle Transport (`bench_modsim_kripke`)
- **What Was Benchmarked**:
  - 3D discrete-ordinates ($S_n$) diamond-difference wavefront sweep (`_kripke_sweep_v6e_kernel`) across `32` spatial zone slabs × `1024` energy groups × `512` angular directions (`16.78M` unknowns/chip, `268.4M` unknowns across 16 chips).
  - Uses `dimension_semantics=("arbitrary",)` to enforce sequential wavefront dependencies across `2 MiB` zone slabs while pinning the upwind angular flux (`scratch_psi_ref`) in on-chip VMEM and overlapping ICI boundary flux exchange.
  - **Achieved Performance**: **`1,173.13 GUnknowns/s` (`1.17 TUnknowns/s`)** (`0.23 ms` step latency, `879.8 GB/s/chip` = **53.7% HBM**, `293.3 GB/s` 16-chip ICI bandwidth).
- **Use Case**:
  - **Deterministic Neutron & Photon Radiation Transport**: Solves the linear Boltzmann transport equation for nuclear reactor physics, radiation shielding, and inertial confinement fusion target energetics.

### 6. ModSim: Branson (Monte Carlo) & SPARTA (DSMC) (`bench_modsim_branson_sparta`)
- **What Was Benchmarked**:
  - Stochastic particle push and distance-to-collision tracking (`_branson_sparta_v6e_kernel`) across `33.55M` particles per chip (`536.8M` particles across 16 chips) using `1024×1024` (`4 MiB`) VMEM super-tiles.
  - Replaces divergent per-particle branch conditionals (`if step < d_coll`) with branchless SIMD `jnp.minimum(step, d_coll)` across 1,024-wide VPU lanes while concurrently overlapping cross-chip particle migration (`lax.ppermute`) over ICI.
  - **Achieved Performance**: **`1,122.01 GParticles/s` (`1.12 TParticles/s`)** (`0.48 ms` step latency, `1,122.0 GB/s/chip` = **68.5% HBM**, `280.5 GB/s` 16-chip ICI bandwidth).
- **Use Case**:
  - **Stochastic Thermal Radiation & Rarefied Gas Dynamics**:
    - **Branson**: Implicit Monte Carlo (IMC) photon transport in high-energy-density plasmas.
    - **SPARTA** (Direct Simulation Monte Carlo): Hypersonic reentry, upper-atmosphere rarefied gas dynamics, and plasma-surface interactions.

### 7. ModSim: AMG2023 Algebraic Multigrid Solver (`bench_modsim_amg2023`)
- **What Was Benchmarked**:
  - Fused 3-step Chebyshev polynomial smoother + Block-CSR Sparse Matrix-Vector Multiply (SpMV) (`_amg2023_v6e_kernel`) over `16.78M` unknowns per chip (`268.4M` unknowns across 16 chips) using `2048×256` (`2 MiB`) VMEM tiles.
  - Reformulates block-sparse coupling operators into `256×256` dense blocks on the MXU and fuses 3 Chebyshev smoothing iterations ($x^{(1)}, x^{(2)}, x^{(3)}$) inside a single VMEM residency pass to cut HBM traffic by $3\times$.
  - **Achieved Performance**: **`47,851.23 GB/s` (`47.85 TB/s`) Effective 3-Step SpMV BW** (`0.27 ms` step latency, `996.9 GB/s/chip` = **60.9% physical HBM**, `95.7 TFLOP/s/chip` = **10.4% MXU**).
- **Use Case**:
  - **Implicit Linear Solvers for Unstructured Multi-Physics**: Solves large sparse linear systems ($Ax = b$) arising from implicit radiation diffusion, incompressible fluid flow, and structural mechanics at extreme scale.

### 8. Collectives: MPI — OMB / Phloem / Sandia ISx (`bench_collectives_omb_isx`)
- **What Was Benchmarked**:
  - A 16-chip global `lax.all_to_all` bucket exchange across the `4x4` 2D Torus ICI interconnect followed by a VMEM-tiled vectorized radix/bitonic sort (`lax.sort` over 128-lane VREG sub-buckets) on `67.11M` 32-bit integer keys per chip (`1.07 billion` keys / `4.29 GB` globally).
  - **Achieved Performance**: **`521.46 GB/s` Global Sort+Exch** (`8.24 ms` step latency, **`977.7 GB/s` 16-chip ICI bandwidth** = `61.1 GB/s/chip`).
- **Use Case**:
  - **Bisection Bandwidth & Distributed Data Redistribution**:
    - **OSU Micro-Benchmarks (OMB) & Phloem**: Measures multi-host ICI bisection bandwidth and all-to-all collective congestion.
    - **Sandia ISx (Integer Sort)**: Models global particle re-binning, adaptive mesh refinement (AMR) load balancing, and distributed graph partitioning.

### 9. Storage: IO500 & MLPerf Storage Checkpoint Stream (`bench_storage_io500`)
- **What Was Benchmarked**:
  - High-throughput streaming DMA read/transform/writeback pipeline (`_storage_v6e_kernel`) over `256 MiB` input + `256 MiB` output per chip (`8.0 GiB` per step across 16 chips) using `2048×1024` (`8 MiB`) double-buffered VMEM super-tiles (`32 MiB` active VMEM).
  - **Achieved Performance**: **`19,132.41 GB/s` (`19.13 TB/s`) Aggregate DMA BW** (`0.45 ms` step latency, `1,195.8 GB/s/chip` = **73.0%** of raw HBM2e peak bandwidth).
- **Use Case**:
  - **Defensive Simulation Checkpointing & AI Training Data Ingest**:
    - **IO500**: High-frequency defensive state checkpoint/restart for long-running hydro/transport runs.
    - **MLPerf Storage (DLIO)**: Sustaining multi-gigabyte/s training sample ingestion and model weight checkpointing without stalling TPU compute engines.

---

## Part 2: Complete 16-Chip Cloud TPU v6e Hardware Telemetry Tables

### Table 1: Compute Units (`MXU` & `VPU/XLU`) & Overall Figure of Merit (FOM)

| Workload | Step Time (ms) | Compile (ms) | Primary Figure of Merit (FOM) | MXU (TFLOP/s/chip) | MXU Util (%) | VPU/XLU (TFLOP/s/chip) | VPU Util (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | **1.11** | 3.7 | **7,183.79 TFLOP/s** | 448.2 | 48.8% | 0.75 | 2.6% |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | **0.23** | 6.3 | **2,378.67 GDOFs/s** | 228.4 | 24.9% | 0.30 | 1.0% |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | **0.19** | 6.5 | **5,436.61 MAtom-Steps/s** | 178.1 | 19.4% | 1.22 | 4.2% |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | **0.92** | 9.3 | **18,690.29 GB/s Stream BW** | 0.0 | 0.0% | 0.66 | 2.3% |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | **0.23** | 5.0 | **1,173.13 GUnknowns/s** | 0.0 | 0.0% | 0.51 | 1.8% |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | **0.48** | 2.5 | **1,122.01 GParticles/s** | 0.0 | 0.0% | 0.42 | 1.5% |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | **0.27** | 2.0 | **47,851.23 GB/s Effective 3-Step SpMV BW** | 95.7 | 10.4% | 1.12 | 3.9% |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | **8.24** | 51.5 | **521.46 GB/s Global Sort+Exch** | 0.0 | 0.0% | 0.23 | 0.8% |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | **0.45** | 1.9 | **19,132.41 GB/s Aggregate DMA BW** | 0.0 | 0.0% | 0.30 | 1.0% |

### Table 2: Memory Hierarchy (`HBM`, `VMEM`, `SMEM`, `SPMEM`) Usage & Bandwidth

| Workload | HBM Base / Peak (MB) | HBM Overhead | HBM BW / Chip (GB/s) | HBM Util (%) | VMEM Used (KiB / %) | VMEM BW (TB/s) | SMEM (KiB) | SparseCore SPMEM (KiB / %) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | 140 / 284 | **2.03x** | 151.9 | 9.3% | 32768 (25.0%) | 0.42 | 64 | 0 (0.0%) |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | 96 / 227 | **2.37x** | 743.3 | 45.4% | 20480 (15.6%) | 4.16 | 32 | 0 (0.0%) |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | 64 / 193 | **3.02x** | 695.9 | 42.5% | 18432 (14.1%) | 3.48 | 32 | 128 (25.0%) |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | 768 / 1282 | **1.67x** | 1168.1 | 71.3% | 32768 (25.0%) | 2.34 | 16 | 0 (0.0%) |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | 128 / 261 | **2.04x** | 879.8 | 53.7% | 14336 (10.9%) | 1.41 | 24 | 0 (0.0%) |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | 384 / 649 | **1.69x** | 1122.0 | 68.5% | 32768 (25.0%) | 2.24 | 48 | 384 (75.0%) |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | 192 / 322 | **1.68x** | 996.9 | 60.9% | 20480 (15.6%) | 2.99 | 32 | 256 (50.0%) |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | 256 / 1024 | **4.00x** | 65.2 | 4.0% | 65536 (50.0%) | 0.23 | 128 | 448 (87.5%) |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | 256 / 770 | **3.01x** | 1195.8 | 73.0% | 32768 (25.0%) | 1.20 | 16 | 0 (0.0%) |

### Table 3: Data Movement Engines (`DMA` & `2D Torus ICI`) & `.xplane.pb` Trace Files

| Workload | DMA Volume / Chip (MB) | DMA Throughput / Chip (GB/s) | Compute/DMA Overlap (%) | ICI Volume / Chip (MB) | ICI BW / Chip (GB/s) | 16-Chip ICI BW (GB/s) | ICI Util (%) | `.xplane.pb` Files |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | 185.0 | 174.5 | **95.2%** | 24.0 | 22.6 | **362.3** | 3.1% | 1 (`ai4sim_llama_scaffold/`) |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | 161.0 | 748.0 | **96.9%** | 1.0 | 4.6 | **74.3** | 0.6% | 1 (`modsim_fem_laghos/`) |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | 128.0 | 695.9 | **96.5%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`modsim_lammps_ace/`) |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | 1024.0 | 1168.1 | **99.2%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`modsim_raja_hydro/`) |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | 196.0 | 898.2 | **98.1%** | 4.0 | 18.3 | **293.3** | 2.5% | 1 (`modsim_kripke_sn/`) |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | 520.0 | 1139.5 | **98.4%** | 8.0 | 17.5 | **280.5** | 2.4% | 1 (`modsim_branson_sparta/`) |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | 257.0 | 1000.8 | **98.0%** | 1.0 | 3.9 | **62.3** | 0.5% | 1 (`modsim_amg2023/`) |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | 992.0 | 126.3 | **96.5%** | 480.0 | 61.1 | **977.7** | 8.5% | 1 (`collectives_omb_isx/`) |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | 512.0 | 1195.8 | **98.4%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`storage_io500_mlperf/`) |
