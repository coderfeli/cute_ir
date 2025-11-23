//===- cute_runtime.h - CuTe Runtime Library Interface ----------*- C++ -*-===//
//
// Runtime library for executing CuTe compiled kernels
//
//===----------------------------------------------------------------------===//

#ifndef CUTE_RUNTIME_H
#define CUTE_RUNTIME_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace cute {
namespace runtime {

//===----------------------------------------------------------------------===//
// Error Handling
//===----------------------------------------------------------------------===//

class CuteRuntimeError : public std::runtime_error {
public:
    explicit CuteRuntimeError(const std::string& msg) 
        : std::runtime_error(msg) {}
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw CuteRuntimeError(std::string("CUDA Error: ") + \
                                   cudaGetErrorString(err)); \
        } \
    } while(0)

#define CU_CHECK(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(err, &errStr); \
            throw CuteRuntimeError(std::string("CUDA Driver Error: ") + \
                                   errStr); \
        } \
    } while(0)

//===----------------------------------------------------------------------===//
// Device Memory Management
//===----------------------------------------------------------------------===//

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0) {}
    
    explicit DeviceBuffer(size_t num_elements) : size_(num_elements) {
        CUDA_CHECK(cudaMalloc(&ptr_, num_elements * sizeof(T)));
    }
    
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    
    // Move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Disable copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Host ↔ Device transfer
    void copy_from_host(const T* host_ptr, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, count * sizeof(T), 
                              cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(T* host_ptr, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, count * sizeof(T), 
                              cudaMemcpyDeviceToHost));
    }
    
    void* ptr() { return ptr_; }
    const void* ptr() const { return ptr_; }
    size_t size() const { return size_; }
    
private:
    void* ptr_;
    size_t size_;
};

//===----------------------------------------------------------------------===//
// TMA Descriptor Manager (SM90+)
//===----------------------------------------------------------------------===//

enum class SwizzleMode {
    None = 0,
    Swizzle32B = 1,
    Swizzle64B = 2,
    Swizzle128B = 3
};

class TMADescriptor {
public:
    TMADescriptor() : desc_(nullptr) {
        CUDA_CHECK(cudaMalloc(&desc_, sizeof(CUtensorMap)));
    }
    
    ~TMADescriptor() {
        if (desc_) cudaFree(desc_);
    }
    
    // Initialize for 2D tensor
    void initialize_2d(
        void* global_ptr,
        cudaDataType dtype,
        uint32_t global_dim_x,
        uint32_t global_dim_y,
        uint32_t tile_dim_x,
        uint32_t tile_dim_y,
        SwizzleMode swizzle = SwizzleMode::Swizzle128B
    );
    
    CUtensorMap* get() { return desc_; }
    
private:
    CUtensorMap* desc_;
};

//===----------------------------------------------------------------------===//
// Kernel Executor
//===----------------------------------------------------------------------===//

struct LaunchConfig {
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem_bytes = 0;
    cudaStream_t stream = nullptr;
    
    LaunchConfig(dim3 grid, dim3 block, size_t smem = 0)
        : grid_dim(grid), block_dim(block), shared_mem_bytes(smem) {}
};

class KernelExecutor {
public:
    KernelExecutor();
    ~KernelExecutor();
    
    // Load compiled kernel from cubin/ptx
    void load_cubin(const std::string& cubin_path);
    void load_ptx(const std::string& ptx_path);
    
    // Set kernel function to execute
    void set_kernel(const std::string& kernel_name);
    
    // Launch kernel with arguments
    void launch(const std::vector<void*>& args, const LaunchConfig& config);
    
    // Synchronize
    void synchronize();
    
    // Get device properties
    static cudaDeviceProp get_device_properties(int device_id = 0);
    
private:
    CUmodule module_;
    CUfunction kernel_;
    bool module_loaded_;
    bool kernel_set_;
};

//===----------------------------------------------------------------------===//
// High-Level GEMM Interface
//===----------------------------------------------------------------------===//

enum class Arch {
    SM80 = 80,  // Ampere
    SM90 = 90,  // Hopper
    SM100 = 100 // Blackwell
};

template<typename TA, typename TB, typename TC>
class GemmExecutor {
public:
    GemmExecutor(
        size_t M, size_t N, size_t K,
        Arch arch = Arch::SM90,
        bool use_tma = true
    );
    
    // Compile from MLIR source
    void compile_from_mlir(const std::string& mlir_code);
    
    // Load pre-compiled kernel
    void load_compiled(const std::string& cubin_path);
    
    // Execute GEMM: C = A @ B
    void execute(
        const TA* A,  // Host or device pointer
        const TB* B,
        TC* C,
        bool is_device_ptr = false
    );
    
    // Get recommended tile sizes
    static std::tuple<size_t, size_t, size_t> get_optimal_tile_size(
        size_t M, size_t N, size_t K, Arch arch
    );
    
private:
    size_t M_, N_, K_;
    Arch arch_;
    bool use_tma_;
    
    std::unique_ptr<KernelExecutor> executor_;
    std::unique_ptr<DeviceBuffer<TA>> d_A_;
    std::unique_ptr<DeviceBuffer<TB>> d_B_;
    std::unique_ptr<DeviceBuffer<TC>> d_C_;
    std::unique_ptr<TMADescriptor> tma_desc_A_;
    std::unique_ptr<TMADescriptor> tma_desc_B_;
};

//===----------------------------------------------------------------------===//
// Compiler Interface
//===----------------------------------------------------------------------===//

class CuteCompiler {
public:
    CuteCompiler();
    
    // Set MLIR tools path
    void set_mlir_bin_path(const std::string& path);
    
    // Compile MLIR to PTX
    std::string compile_to_ptx(
        const std::string& mlir_code,
        Arch arch,
        int opt_level = 2
    );
    
    // Compile PTX to CUBIN
    std::string compile_to_cubin(
        const std::string& ptx_code,
        Arch arch
    );
    
    // Full compilation: MLIR → CUBIN
    std::string compile(
        const std::string& mlir_code,
        Arch arch,
        int opt_level = 2
    );
    
private:
    std::string mlir_bin_path_;
    
    std::string run_command(const std::string& cmd);
};

} // namespace runtime
} // namespace cute

#endif // CUTE_RUNTIME_H
