#pragma once

#include <limits>

// torch
#include <ATen/core/Array.h>
#include <ATen/TensorIterator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/ops/empty.h>

// disort
#include <cdisort213/pmem.h>

namespace disort {
namespace native {

template <typename func_t>
__global__ void element_kernel(int64_t numel, func_t f) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    f(idx);
  }
}

// Chunked variant for element functions that need a per-thread work pool.
// The iteration range is split only
// when one workspace of work_size bytes per *concurrent* thread would exceed
// half of the free device memory.  A DisortImpl-owned workspace is grown on
// demand and reused across forwards.  The workspace is bound to this
// translation unit's pmem per-thread pool globals; the element lambda is
// expected to call pmem::pool_init() before its first pmalloc.
template <int Arity, typename func_t>
void gpu_chunk_kernel(at::TensorIterator& iter, size_t work_size,
                      at::Tensor* workspace_cache, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = reinterpret_cast<char*>(iter.data_ptr(i));
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();
  if (numel == 0) return;

  TORCH_CHECK(workspace_cache != nullptr,
              "gpu_chunk_kernel: CUDA workspace cache is null");
  int64_t cached_elem = 0;
  if (workspace_cache->defined() &&
      workspace_cache->device() == iter.device()) {
    cached_elem = static_cast<int64_t>(workspace_cache->numel() / work_size);
  }

  // Grow the instance-owned cache only when the current batch needs more
  // concurrent solver slices.  Smaller batches reuse the existing storage.
  if (cached_elem < numel) {
    size_t mem_free = 0, mem_total = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&mem_free, &mem_total));
    int64_t max_elem = static_cast<int64_t>((mem_free / 2) / work_size);
    TORCH_CHECK(max_elem > 0, "gpu_chunk_kernel: per-thread work size (",
                work_size, " B) exceeds half of free device memory");
    int64_t target_elem = std::min(numel, max_elem);
    TORCH_CHECK(target_elem <=
                    std::numeric_limits<int64_t>::max() /
                        static_cast<int64_t>(work_size),
                "gpu_chunk_kernel: workspace size overflows int64_t");
    *workspace_cache = at::empty(
        {target_elem * static_cast<int64_t>(work_size)},
        at::TensorOptions().device(iter.device()).dtype(at::kByte));
    cached_elem = target_elem;
  }

  int64_t chunks = (numel + cached_elem - 1) / cached_elem;
  int64_t base = numel / chunks;
  int64_t rem = numel % chunks;
  char* workspace = static_cast<char*>(workspace_cache->data_ptr());
  cudaError_t err = pmem::bind_workspace(workspace, work_size);
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "gpu_chunk_kernel: binding pmem workspace failed: ",
                cudaGetErrorString(err));
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t chunk_start = 0;
  for (int64_t c = 0; c < chunks; c++) {
    int64_t chunk_numel = base + (c < rem ? 1 : 0);

    auto device_lambda = [=] __device__(int idx) {
      auto offsets = offset_calc.get((int)(idx + chunk_start));
      f(data.data(), offsets.data());
    };

    // Each element is an independent, register-heavy scalar DISORT solve.
    // A single warp per block exposes more blocks to the scheduler for the
    // common GCM case where nwave * ncol is only a few hundred.
    dim3 block(32);
    dim3 grid((unsigned)((chunk_numel + block.x - 1) / block.x));
    element_kernel<<<grid, block, 0, stream>>>(chunk_numel, device_lambda);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // the workspace is reused by the next chunk
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    chunk_start += chunk_numel;
  }
}

}  // namespace native
}  // namespace disort
