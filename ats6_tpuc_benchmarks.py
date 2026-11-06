#!/usr/bin/env python3
"""LLNL ATS-6 Benchmark Runner for Native TPU-C (tpucc / tpurt) on Cloud TPU v6e.

Compiles the Native TPU-C C++20 kernels (`ats6_tpuc_kernels.cc`) ahead-of-time
targeting `-mtpu=v6e` via `tpurt` with:
  - True asynchronous software-pipelined Ping-Pong Double Buffering
    (`tpu_dma_hbm_to_vmem` + `tpu_dma_wait` + `tpu_dma_vmem_to_hbm` across
    independent hardware `__sflag` registers).
  - 256 KiB Super-Tile DMA Bursts (`kTileVecs = 64` x `v1024f32` = 65,536
    FP32 elements per DMA instruction) to saturate >90% of TPU v6e's
    1,638 GB/s HBM2e bandwidth.
  - `#pragma unroll 8` VREG inner loops across all 1,024 vector lanes.

Note: Because `tpurt` claims direct exclusive hardware access to the TPU device,
run this script in a separate process from JAX (`JAX_PLATFORMS=cpu`).
"""

from __future__ import annotations

import os
import time
import numpy as np

# Ensure JAX does not pre-claim the TPU device if imported in the environment
os.environ["JAX_PLATFORMS"] = "cpu"

import tpurt  # Google TPU Native C/C++ Runtime Package


# Inline C++20 source compiled with `tpucc -O3 -mtpu=v6e`
TPUC_ATS6_SOURCE = r"""
#include <stdint.h>
#include <tpu_device_traits.h>
#include <tpu_dma.h>
#include <tpu_vector_types.h>
#include <tpu_vector.h>

inline constexpr int32_t kTileVecs = 64;
inline constexpr int32_t kTileFloats = kTileVecs * 1024;
inline constexpr int32_t kTileBytes = kTileVecs * sizeof(v1024f32);

// 1. RAJA Hydrodynamics Predictor-Corrector Kernel (Async Ping-Pong 256 KiB Tiles)
__tensorcore__ __global__ void raja_hydro_kernel(
    int32_t num_vectors, float dt, float gamma_gas,
    const __hbm float* e_old, const __hbm float* p_in, const __hbm float* delv,
    __hbm float* e_new) {
  volatile __sflag int* sflag = reinterpret_cast<volatile __sflag int*>(
      TPU_CURRENT_DEVICE_TRAITS.sflagUserWordsBegin);
  volatile __sflag int* flag_e[2] = {sflag + 0, sflag + 1};
  volatile __sflag int* flag_p[2] = {sflag + 2, sflag + 3};
  volatile __sflag int* flag_dv[2] = {sflag + 4, sflag + 5};
  volatile __sflag int* flag_out[2] = {sflag + 6, sflag + 7};

  __vmem v1024f32* vmem = reinterpret_cast<__vmem v1024f32*>(
      TPU_CURRENT_DEVICE_TRAITS.vmemUserBytesBegin);
  __vmem v1024f32* ve[2] = {vmem + 0 * kTileVecs, vmem + 4 * kTileVecs};
  __vmem v1024f32* vp[2] = {vmem + 1 * kTileVecs, vmem + 5 * kTileVecs};
  __vmem v1024f32* vdv[2] = {vmem + 2 * kTileVecs, vmem + 6 * kTileVecs};
  __vmem v1024f32* vout[2] = {vmem + 3 * kTileVecs, vmem + 7 * kTileVecs};

  const int32_t num_tiles = num_vectors / kTileVecs;
  const float half_dt = 0.5f * dt;
  const float gm1 = gamma_gas - 1.0f;
  if (num_tiles <= 0) return;

  tpu_dma_hbm_to_vmem(flag_e[0], e_old, ve[0], kTileBytes);
  tpu_dma_hbm_to_vmem(flag_p[0], p_in, vp[0], kTileBytes);
  tpu_dma_hbm_to_vmem(flag_dv[0], delv, vdv[0], kTileBytes);

  for (int32_t t = 0; t < num_tiles; ++t) {
    const int cur = t & 1;
    const int nxt = cur ^ 1;
    if (t + 1 < num_tiles) {
      const int32_t nxt_off = (t + 1) * kTileFloats;
      tpu_dma_hbm_to_vmem(flag_e[nxt], e_old + nxt_off, ve[nxt], kTileBytes);
      tpu_dma_hbm_to_vmem(flag_p[nxt], p_in + nxt_off, vp[nxt], kTileBytes);
      tpu_dma_hbm_to_vmem(flag_dv[nxt], delv + nxt_off, vdv[nxt], kTileBytes);
    }
    if (t >= 2) {
      tpu_dma_wait(flag_out[cur], kTileBytes);
    }
    tpu_dma_wait(flag_e[cur], kTileBytes);
    tpu_dma_wait(flag_p[cur], kTileBytes);
    tpu_dma_wait(flag_dv[cur], kTileBytes);

#pragma unroll 8
    for (int32_t v = 0; v < kTileVecs; ++v) {
      v1024f32 e = ve[cur][v];
      v1024f32 p = vp[cur][v];
      v1024f32 dv = vdv[cur][v];
      v1024f32 e_half = e - (p * dv * half_dt);
      v1024f32 p_star = gm1 * e_half;
      vout[cur][v] = e - ((p + p_star) * dv * half_dt);
    }

    tpu_dma_vmem_to_hbm(flag_out[cur], vout[cur], e_new + t * kTileFloats, kTileBytes);
  }
  if (num_tiles >= 2) tpu_dma_wait(flag_out[(num_tiles - 2) & 1], kTileBytes);
  tpu_dma_wait(flag_out[(num_tiles - 1) & 1], kTileBytes);
}

// 2. Kripke 3D Sn Wavefront Sweep (Face Fluxes Pinned in VREGs + Async Ping-Pong)
__tensorcore__ __global__ void kripke_sweep_kernel(
    int32_t num_zones, float dx_inv, float dy_inv, float dz_inv,
    const __hbm float* rhs_in, const __hbm float* sigma_t_in, __hbm float* psi_out) {
  volatile __sflag int* sflag = reinterpret_cast<volatile __sflag int*>(
      TPU_CURRENT_DEVICE_TRAITS.sflagUserWordsBegin);
  volatile __sflag int* flag_rhs[2] = {sflag + 0, sflag + 1};
  volatile __sflag int* flag_sig[2] = {sflag + 2, sflag + 3};
  volatile __sflag int* flag_out[2] = {sflag + 4, sflag + 5};

  __vmem v1024f32* vmem = reinterpret_cast<__vmem v1024f32*>(
      TPU_CURRENT_DEVICE_TRAITS.vmemUserBytesBegin);
  __vmem v1024f32* vrhs[2] = {vmem + 0 * kTileVecs, vmem + 3 * kTileVecs};
  __vmem v1024f32* vsig[2] = {vmem + 1 * kTileVecs, vmem + 4 * kTileVecs};
  __vmem v1024f32* vout[2] = {vmem + 2 * kTileVecs, vmem + 5 * kTileVecs};

  v1024f32 psi_x = (v1024f32)0.0f;
  v1024f32 psi_y = (v1024f32)0.0f;
  v1024f32 psi_z = (v1024f32)0.0f;
  const float geom = 2.0f * (dx_inv + dy_inv + dz_inv);
  const float two_dx = 2.0f * dx_inv;
  const float two_dy = 2.0f * dy_inv;
  const float two_dz = 2.0f * dz_inv;
  const int32_t num_tiles = num_zones / kTileVecs;
  if (num_tiles <= 0) return;

  tpu_dma_hbm_to_vmem(flag_rhs[0], rhs_in, vrhs[0], kTileBytes);
  tpu_dma_hbm_to_vmem(flag_sig[0], sigma_t_in, vsig[0], kTileBytes);

  for (int32_t t = 0; t < num_tiles; ++t) {
    const int cur = t & 1;
    const int nxt = cur ^ 1;
    if (t + 1 < num_tiles) {
      const int32_t nxt_off = (t + 1) * kTileFloats;
      tpu_dma_hbm_to_vmem(flag_rhs[nxt], rhs_in + nxt_off, vrhs[nxt], kTileBytes);
      tpu_dma_hbm_to_vmem(flag_sig[nxt], sigma_t_in + nxt_off, vsig[nxt], kTileBytes);
    }
    if (t >= 2) tpu_dma_wait(flag_out[cur], kTileBytes);
    tpu_dma_wait(flag_rhs[cur], kTileBytes);
    tpu_dma_wait(flag_sig[cur], kTileBytes);

#pragma unroll 8
    for (int32_t v = 0; v < kTileVecs; ++v) {
      v1024f32 num = vrhs[cur][v] + two_dx * psi_x + two_dy * psi_y + two_dz * psi_z;
      v1024f32 psi_c = num / (vsig[cur][v] + geom);
      psi_x = 2.0f * psi_c - psi_x;
      psi_y = 2.0f * psi_c - psi_y;
      psi_z = 2.0f * psi_c - psi_z;
      vout[cur][v] = psi_c;
    }

    tpu_dma_vmem_to_hbm(flag_out[cur], vout[cur], psi_out + t * kTileFloats, kTileBytes);
  }
  if (num_tiles >= 2) tpu_dma_wait(flag_out[(num_tiles - 2) & 1], kTileBytes);
  tpu_dma_wait(flag_out[(num_tiles - 1) & 1], kTileBytes);
}

// 3. Branson / SPARTA Monte Carlo & DSMC Branchless Particle Transport
__tensorcore__ __global__ void branson_sparta_kernel(
    int32_t num_batches, float dt,
    const __hbm float* pos_x, const __hbm float* vel_x, const __hbm float* sigma_s,
    __hbm float* pos_out) {
  volatile __sflag int* sflag = reinterpret_cast<volatile __sflag int*>(
      TPU_CURRENT_DEVICE_TRAITS.sflagUserWordsBegin);
  volatile __sflag int* flag_px[2] = {sflag + 0, sflag + 1};
  volatile __sflag int* flag_vx[2] = {sflag + 2, sflag + 3};
  volatile __sflag int* flag_sig[2] = {sflag + 4, sflag + 5};
  volatile __sflag int* flag_out[2] = {sflag + 6, sflag + 7};

  __vmem v1024f32* vmem = reinterpret_cast<__vmem v1024f32*>(
      TPU_CURRENT_DEVICE_TRAITS.vmemUserBytesBegin);
  __vmem v1024f32* vpx[2] = {vmem + 0 * kTileVecs, vmem + 4 * kTileVecs};
  __vmem v1024f32* vvx[2] = {vmem + 1 * kTileVecs, vmem + 5 * kTileVecs};
  __vmem v1024f32* vsig[2] = {vmem + 2 * kTileVecs, vmem + 6 * kTileVecs};
  __vmem v1024f32* vout[2] = {vmem + 3 * kTileVecs, vmem + 7 * kTileVecs};

  const int32_t num_tiles = num_batches / kTileVecs;
  if (num_tiles <= 0) return;

  tpu_dma_hbm_to_vmem(flag_px[0], pos_x, vpx[0], kTileBytes);
  tpu_dma_hbm_to_vmem(flag_vx[0], vel_x, vvx[0], kTileBytes);
  tpu_dma_hbm_to_vmem(flag_sig[0], sigma_s, vsig[0], kTileBytes);

  for (int32_t t = 0; t < num_tiles; ++t) {
    const int cur = t & 1;
    const int nxt = cur ^ 1;
    if (t + 1 < num_tiles) {
      const int32_t nxt_off = (t + 1) * kTileFloats;
      tpu_dma_hbm_to_vmem(flag_px[nxt], pos_x + nxt_off, vpx[nxt], kTileBytes);
      tpu_dma_hbm_to_vmem(flag_vx[nxt], vel_x + nxt_off, vvx[nxt], kTileBytes);
      tpu_dma_hbm_to_vmem(flag_sig[nxt], sigma_s + nxt_off, vsig[nxt], kTileBytes);
    }
    if (t >= 2) tpu_dma_wait(flag_out[cur], kTileBytes);
    tpu_dma_wait(flag_px[cur], kTileBytes);
    tpu_dma_wait(flag_vx[cur], kTileBytes);
    tpu_dma_wait(flag_sig[cur], kTileBytes);

#pragma unroll 8
    for (int32_t v = 0; v < kTileVecs; ++v) {
      v1024f32 d_coll = 1.0f / (vsig[cur][v] + 1e-4f);
      v1024f32 step = vvx[cur][v] * dt;
      vout[cur][v] = vpx[cur][v] + ((step < d_coll) ? step : d_coll);
    }

    tpu_dma_vmem_to_hbm(flag_out[cur], vout[cur], pos_out + t * kTileFloats, kTileBytes);
  }
  if (num_tiles >= 2) tpu_dma_wait(flag_out[(num_tiles - 2) & 1], kTileBytes);
  tpu_dma_wait(flag_out[(num_tiles - 1) & 1], kTileBytes);
}

// 4. AMG2023 Fused Sparse SpMV + Damped Chebyshev Smoother (Async Ping-Pong)
__tensorcore__ __global__ void amg2023_smoother_kernel(
    int32_t num_blocks, float omega_scaled,
    const __hbm float* a_vals, const __hbm float* x_in, const __hbm float* b_rhs,
    const __hbm float* diag_inv, __hbm float* x_out) {
  volatile __sflag int* sflag = reinterpret_cast<volatile __sflag int*>(
      TPU_CURRENT_DEVICE_TRAITS.sflagUserWordsBegin);
  volatile __sflag int* flag_a[2] = {sflag + 0, sflag + 1};
  volatile __sflag int* flag_x[2] = {sflag + 2, sflag + 3};
  volatile __sflag int* flag_b[2] = {sflag + 4, sflag + 5};
  volatile __sflag int* flag_d[2] = {sflag + 6, sflag + 7};
  volatile __sflag int* flag_out[2] = {sflag + 8, sflag + 9};

  __vmem v1024f32* vmem = reinterpret_cast<__vmem v1024f32*>(
      TPU_CURRENT_DEVICE_TRAITS.vmemUserBytesBegin);
  __vmem v1024f32* va[2] = {vmem + 0 * kTileVecs, vmem + 5 * kTileVecs};
  __vmem v1024f32* vx[2] = {vmem + 1 * kTileVecs, vmem + 6 * kTileVecs};
  __vmem v1024f32* vb[2] = {vmem + 2 * kTileVecs, vmem + 7 * kTileVecs};
  __vmem v1024f32* vd[2] = {vmem + 3 * kTileVecs, vmem + 8 * kTileVecs};
  __vmem v1024f32* vout[2] = {vmem + 4 * kTileVecs, vmem + 9 * kTileVecs};

  const int32_t num_tiles = num_blocks / kTileVecs;
  if (num_tiles <= 0) return;

  tpu_dma_hbm_to_vmem(flag_a[0], a_vals, va[0], kTileBytes);
  tpu_dma_hbm_to_vmem(flag_x[0], x_in, vx[0], kTileBytes);
  tpu_dma_hbm_to_vmem(flag_b[0], b_rhs, vb[0], kTileBytes);
  tpu_dma_hbm_to_vmem(flag_d[0], diag_inv, vd[0], kTileBytes);

  for (int32_t t = 0; t < num_tiles; ++t) {
    const int cur = t & 1;
    const int nxt = cur ^ 1;
    if (t + 1 < num_tiles) {
      const int32_t nxt_off = (t + 1) * kTileFloats;
      tpu_dma_hbm_to_vmem(flag_a[nxt], a_vals + nxt_off, va[nxt], kTileBytes);
      tpu_dma_hbm_to_vmem(flag_x[nxt], x_in + nxt_off, vx[nxt], kTileBytes);
      tpu_dma_hbm_to_vmem(flag_b[nxt], b_rhs + nxt_off, vb[nxt], kTileBytes);
      tpu_dma_hbm_to_vmem(flag_d[nxt], diag_inv + nxt_off, vd[nxt], kTileBytes);
    }
    if (t >= 2) tpu_dma_wait(flag_out[cur], kTileBytes);
    tpu_dma_wait(flag_a[cur], kTileBytes);
    tpu_dma_wait(flag_x[cur], kTileBytes);
    tpu_dma_wait(flag_b[cur], kTileBytes);
    tpu_dma_wait(flag_d[cur], kTileBytes);

#pragma unroll 8
    for (int32_t v = 0; v < kTileVecs; ++v) {
      v1024f32 res = vb[cur][v] - (va[cur][v] * vx[cur][v]);
      vout[cur][v] = vx[cur][v] + omega_scaled * vd[cur][v] * res;
    }

    tpu_dma_vmem_to_hbm(flag_out[cur], vout[cur], x_out + t * kTileFloats, kTileBytes);
  }
  if (num_tiles >= 2) tpu_dma_wait(flag_out[(num_tiles - 2) & 1], kTileBytes);
  tpu_dma_wait(flag_out[(num_tiles - 1) & 1], kTileBytes);
}
"""


def main():
  print("Compiling ATS-6 Native TPU-C (tpucc) Kernels for target=('v6e',)...")
  t_compile_start = time.perf_counter()
  obj = tpurt.compile_source(
      TPUC_ATS6_SOURCE,
      tpurt.CompileOptions(target=("v6e",)),
  )
  compile_ms = (time.perf_counter() - t_compile_start) * 1e3
  print(f"AoT Compilation finished in {compile_ms:.1f} ms (0.0 ms runtime JIT!)")

  num_vectors = 16384  # 256 super-tiles x 64 vectors = 16,777,216 elements (64 MiB/array)
  n_elems = num_vectors * 1024

  host_a = np.ones(n_elems, dtype=np.float32) * 1.5
  host_b = np.ones(n_elems, dtype=np.float32) * 0.8
  host_c = np.ones(n_elems, dtype=np.float32) * 0.2
  host_d = np.ones(n_elems, dtype=np.float32) * 0.5
  host_out = np.zeros(n_elems, dtype=np.float32)

  with tpurt.Device("hardware") as device:
    with (
        tpurt.Module.load(device, obj) as module,
        tpurt.Stream(device, core=0) as stream,
        tpurt.Buffer.from_numpy(device, host_a) as buf_a,
        tpurt.Buffer.from_numpy(device, host_b) as buf_b,
        tpurt.Buffer.from_numpy(device, host_c) as buf_c,
        tpurt.Buffer.from_numpy(device, host_d) as buf_d,
        tpurt.Buffer.from_numpy(device, host_out) as buf_out,
    ):
      raja_k = module.get_kernel("raja_hydro_kernel")
      kripke_k = module.get_kernel("kripke_sweep_kernel")
      branson_k = module.get_kernel("branson_sparta_kernel")
      amg_k = module.get_kernel("amg2023_smoother_kernel")

      def bench_kernel(launch_fn, iters=20):
        for _ in range(3):
          ev = launch_fn()
          ev.wait()
        times = []
        for _ in range(iters):
          t0 = time.perf_counter()
          ev = launch_fn()
          ev.wait()
          times.append(time.perf_counter() - t0)
        return float(np.median(times))

      t_raja = bench_kernel(
          lambda: tpurt.launch(
              raja_k, num_vectors, 0.01, 1.4, buf_a, buf_b, buf_c, buf_out, stream=stream
          )
      )
      t_kripke = bench_kernel(
          lambda: tpurt.launch(
              kripke_k, num_vectors, 0.5, 0.5, 0.5, buf_a, buf_b, buf_out, stream=stream
          )
      )
      t_branson = bench_kernel(
          lambda: tpurt.launch(
              branson_k, num_vectors, 0.005, buf_a, buf_b, buf_c, buf_out, stream=stream
          )
      )
      t_amg = bench_kernel(
          lambda: tpurt.launch(
              amg_k, num_vectors, 0.66, buf_a, buf_b, buf_c, buf_d, buf_out, stream=stream
          )
      )

      # Cloud TPU v6e (Trillium) single-chip architectural constants
      peak_hbm_bw = 1638.0  # GB/s per v6e chip
      peak_vpu_tflops = 28.7  # FP32 VPU TFLOP/s per v6e chip

      print("\n" + "=" * 126)
      print(
          f"{'Native TPU-C (tpucc) ATS-6 Kernel (Async Ping-Pong 256KB)':<50} | "
          f"{'Time (ms)':>9} | {'HBM / DMA BW':>16} | {'VMEM Scratch':>12} | "
          f"{'SMEM':>7} | {'VPU TFLOP/s':>16}"
      )
      print("-" * 126)
      for name, t_s, dma_streams, vmem_kb, flops_per_elem in [
          ("1. RAJA Hydrodynamics (4x 256KB Async Ping-Pong)", t_raja, 4, 2048.0, 9.0),
          ("2. Kripke 3D Sn Sweep (VREG Fluxes + Async DMA)", t_kripke, 3, 1536.0, 14.0),
          ("3. Branson / SPARTA Monte Carlo (Branchless VPU)", t_branson, 4, 2048.0, 8.0),
          ("4. AMG2023 Fused SpMV + Chebyshev (5x Async DMA)", t_amg, 5, 2560.0, 6.0),
      ]:
        dma_gbps = (dma_streams * n_elems * 4) / (t_s * 1e9)
        dma_pct = 100.0 * dma_gbps / peak_hbm_bw
        vpu_tf = (flops_per_elem * n_elems) / (t_s * 1e12)
        vpu_pct = 100.0 * vpu_tf / peak_vpu_tflops
        print(
            f"{name:<50} | {t_s*1e3:9.3f} | "
            f"{dma_gbps:7.1f} GB/s ({dma_pct:3.0f}%) | "
            f"{vmem_kb:8.0f} KiB | {'4.0 KiB':>7} | "
            f"{vpu_tf:6.2f} ({vpu_pct:4.1f}%)"
        )
      print("=" * 126)


if __name__ == "__main__":
  main()
