// HELIX CUDA Runtime Functions
//
// Provides basic CUDA runtime operations for the FFI interface.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

extern "C" {

// ============================================================================
// CUDA Runtime API Wrappers
// ============================================================================

int32_t helix_cuda_init() {
    cudaError_t err = cudaSetDevice(0);
    return (err == cudaSuccess) ? 0 : -1;
}

int32_t helix_cuda_is_available() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0) ? 1 : 0;
}

int32_t helix_cuda_get_device_count() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess) ? deviceCount : 0;
}

int32_t helix_cuda_get_device_info(
    int32_t device,
    char* name,
    int32_t name_len,
    int32_t* compute_major,
    int32_t* compute_minor,
    uint64_t* total_memory,
    int32_t* sm_count,
    int32_t* max_threads,
    int32_t* max_shared_mem,
    int32_t* warp_size
) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);

    if (err != cudaSuccess) {
        return -1;
    }

    strncpy(name, prop.name, name_len - 1);
    name[name_len - 1] = '\0';

    *compute_major = prop.major;
    *compute_minor = prop.minor;
    *total_memory = prop.totalGlobalMem;
    *sm_count = prop.multiProcessorCount;
    *max_threads = prop.maxThreadsPerBlock;
    *max_shared_mem = prop.sharedMemPerBlock;
    *warp_size = prop.warpSize;

    return 0;
}

int32_t helix_cuda_set_device(int32_t device) {
    cudaError_t err = cudaSetDevice(device);
    return (err == cudaSuccess) ? 0 : -1;
}

int32_t helix_cuda_device_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Memory Management
// ============================================================================

int32_t helix_cuda_malloc(uint64_t size, uint64_t* ptr) {
    void* device_ptr = nullptr;
    cudaError_t err = cudaMalloc(&device_ptr, size);

    if (err != cudaSuccess) {
        *ptr = 0;
        return -1;
    }

    *ptr = reinterpret_cast<uint64_t>(device_ptr);
    return 0;
}

int32_t helix_cuda_free(uint64_t ptr) {
    if (ptr == 0) return 0;
    cudaError_t err = cudaFree(reinterpret_cast<void*>(ptr));
    return (err == cudaSuccess) ? 0 : -1;
}

int32_t helix_cuda_memcpy_htod(uint64_t dst, const uint8_t* src, size_t size) {
    cudaError_t err = cudaMemcpy(
        reinterpret_cast<void*>(dst),
        src,
        size,
        cudaMemcpyHostToDevice
    );
    return (err == cudaSuccess) ? 0 : -1;
}

int32_t helix_cuda_memcpy_dtoh(uint8_t* dst, uint64_t src, size_t size) {
    cudaError_t err = cudaMemcpy(
        dst,
        reinterpret_cast<const void*>(src),
        size,
        cudaMemcpyDeviceToHost
    );
    return (err == cudaSuccess) ? 0 : -1;
}

int32_t helix_cuda_memcpy_dtod(uint64_t dst, uint64_t src, size_t size) {
    cudaError_t err = cudaMemcpy(
        reinterpret_cast<void*>(dst),
        reinterpret_cast<const void*>(src),
        size,
        cudaMemcpyDeviceToDevice
    );
    return (err == cudaSuccess) ? 0 : -1;
}

int32_t helix_cuda_memset(uint64_t ptr, int32_t value, size_t size) {
    cudaError_t err = cudaMemset(reinterpret_cast<void*>(ptr), value, size);
    return (err == cudaSuccess) ? 0 : -1;
}

} // extern "C"
