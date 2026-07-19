#pragma once

// torch
#include <ATen/core/Array.h>
#include <ATen/TensorIterator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/Loops.cuh>

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

// Chunked variant for element functions that need a per-thread work pool
// (modeled on pyharp's gpu_chunk_kernel).  The iteration range is split
// into at least `Chunks` sequential launches so that one workspace of
// work_size bytes per *concurrent* thread -- not per element -- is
// enough; more chunks are used when work_size * elements would exceed
// half of the free device memory.  The workspace is bound to this
// translation unit's pmem per-thread pool globals; the element lambda is
// expected to call pmem::pool_init() before its first pmalloc.
template <int Chunks, int Arity, typename func_t>
void gpu_chunk_kernel(at::TensorIterator& iter, size_t work_size,
                      const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = reinterpret_cast<char*>(iter.data_ptr(i));
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();
  if (numel == 0) return;

  // number of chunks: at least Chunks, more if the workspace would not fit
  size_t mem_free = 0, mem_total = 0;
  C10_CUDA_CHECK(cudaMemGetInfo(&mem_free, &mem_total));
  int64_t max_elem = (int64_t)((mem_free / 2) / work_size);
  TORCH_CHECK(max_elem > 0, "gpu_chunk_kernel: per-thread work size (",
              work_size, " B) exceeds half of free device memory");
  int64_t chunks = std::max<int64_t>(Chunks, (numel + max_elem - 1) / max_elem);
  chunks = std::min<int64_t>(chunks, numel);

  int64_t base = numel / chunks;
  int64_t rem = numel % chunks;
  int64_t max_chunk_numel = base + (rem > 0 ? 1 : 0);

  char* workspace = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&workspace, (size_t)max_chunk_numel * work_size));
  cudaError_t err = pmem::bind_workspace(workspace, work_size);
  if (err != cudaSuccess) {
    cudaFree(workspace);
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

    dim3 block(64);
    dim3 grid((unsigned)((chunk_numel + block.x - 1) / block.x));
    element_kernel<<<grid, block, 0, stream>>>(chunk_numel, device_lambda);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // the workspace is reused by the next chunk
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    chunk_start += chunk_numel;
  }

  C10_CUDA_CHECK(cudaFree(workspace));
}

}  // namespace native
}  // namespace disort
