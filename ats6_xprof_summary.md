# LLNL ATS-6 Cloud TPU v6e (`4x4` = 16 Chips) Complete XProf Hardware Telemetry Report

- **Cluster Topology**: `16` Cloud TPU v6e chips (`4` hosts × `4` chips/host, `4x4` 2D Torus)
- **Data Generation**: 100% On-Device In-Memory (`shard_map` + VPU PRNG, zero external datasets)
- **XProf Trace Directory**: `/tmp/ats6_xprof_traces` (`.xplane.pb` hardware traces captured per workload)

## Table 1: Compute Units (`MXU` & `VPU/XLU`) & Overall Figure of Merit (FOM)

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

## Table 2: Memory Hierarchy (`HBM`, `VMEM`, `SMEM`, `SPMEM`) Usage & Bandwidth

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

## Table 3: Data Movement Engines (`DMA` & `2D Torus ICI`) & `.xplane.pb` Trace Files

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
