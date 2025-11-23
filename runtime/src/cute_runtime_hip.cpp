//===- cute_runtime_hip.cpp - CuTe Runtime HIP/ROCm Implementation -*- C++ -*-===//
//
// HIP/ROCm runtime support for AMD GFX942
//
//===----------------------------------------------------------------------===//

#ifdef HAVE_HIP

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cute {
namespace runtime {
namespace hip {

//===----------------------------------------------------------------------===//
// HIP Error Checking
//===----------------------------------------------------------------------===//

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error( \
                std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + ": " + hipGetErrorString(err)); \
        } \
    } while(0)

#define HIPRTC_CHECK(call) \
    do { \
        hiprtcResult err = call; \
        if (err != HIPRTC_SUCCESS) { \
            throw std::runtime_error( \
                std::string("HIPRTC error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__)); \
        } \
    } while(0)

//===----------------------------------------------------------------------===//
// Device Management
//===----------------------------------------------------------------------===//

class HipDevice {
public:
    static HipDevice& instance() {
        static HipDevice device;
        return device;
    }
    
    void initialize() {
        if (initialized_) return;
        
        int device_count;
        HIP_CHECK(hipGetDeviceCount(&device_count));
        
        if (device_count == 0) {
            throw std::runtime_error("No HIP-capable devices found");
        }
        
        // Use first device by default
        HIP_CHECK(hipSetDevice(0));
        
        // Get device properties
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, 0));
        
        device_name_ = props.name;
        gcn_arch_ = props.gcnArchName;
        
        std::cout << "HIP Device: " << device_name_ << std::endl;
        std::cout << "GCN Architecture: " << gcn_arch_ << std::endl;
        std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
        std::cout << "Total Memory: " << (props.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
        std::cout << "Max Threads per Block: " << props.maxThreadsPerBlock << std::endl;
        std::cout << "Wavefront Size: " << props.warpSize << std::endl;
        
        // Verify GFX942 support
        if (gcn_arch_.find("gfx942") != std::string::npos ||
            gcn_arch_.find("gfx941") != std::string::npos ||
            gcn_arch_.find("gfx940") != std::string::npos) {
            std::cout << "GFX942/MI300 architecture detected - MFMA support available" << std::endl;
        }
        
        initialized_ = true;
    }
    
    std::string getDeviceName() const { return device_name_; }
    std::string getGcnArch() const { return gcn_arch_; }

private:
    HipDevice() : initialized_(false) {}
    
    bool initialized_;
    std::string device_name_;
    std::string gcn_arch_;
};

//===----------------------------------------------------------------------===//
// Kernel Executor for HIP
//===----------------------------------------------------------------------===//

class HipKernelExecutor {
public:
    HipKernelExecutor() : module_(nullptr), kernel_(nullptr) {
        HipDevice::instance().initialize();
    }
    
    ~HipKernelExecutor() {
        if (module_) {
            hipModuleUnload(module_);
        }
    }
    
    // Load precompiled code object (HSACO)
    void loadCodeObject(const std::string& hsaco_path) {
        std::ifstream file(hsaco_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open code object: " + hsaco_path);
        }
        
        std::vector<char> hsaco_data(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );
        
        HIP_CHECK(hipModuleLoadData(&module_, hsaco_data.data()));
    }
    
    // Compile HIP source at runtime
    void compileSource(const std::string& source, const std::string& kernel_name,
                       const std::string& target_arch = "gfx942") {
        hiprtcProgram prog;
        HIPRTC_CHECK(hiprtcCreateProgram(&prog, source.c_str(), 
                                         "kernel.hip", 0, nullptr, nullptr));
        
        // Compilation options for GFX942
        std::vector<const char*> opts = {
            "--offload-arch=gfx942",
            "-O3",
            "-std=c++17"
        };
        
        hiprtcResult compileResult = hiprtcCompileProgram(prog, opts.size(), opts.data());
        
        // Get compilation log
        size_t log_size;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &log_size));
        if (log_size > 1) {
            std::vector<char> log(log_size);
            HIPRTC_CHECK(hiprtcGetProgramLog(prog, log.data()));
            std::cout << "Compilation log:\n" << log.data() << std::endl;
        }
        
        if (compileResult != HIPRTC_SUCCESS) {
            hiprtcDestroyProgram(&prog);
            throw std::runtime_error("Kernel compilation failed");
        }
        
        // Get code object
        size_t code_size;
        HIPRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));
        
        std::vector<char> code(code_size);
        HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
        
        // Load module
        HIP_CHECK(hipModuleLoadData(&module_, code.data()));
        
        // Get kernel function
        HIP_CHECK(hipModuleGetFunction(&kernel_, module_, kernel_name.c_str()));
        
        hiprtcDestroyProgram(&prog);
    }
    
    // Launch kernel
    template<typename... Args>
    void launch(dim3 grid, dim3 block, size_t shared_mem, Args... args) {
        if (!kernel_) {
            throw std::runtime_error("Kernel not loaded");
        }
        
        void* kernel_args[] = { &args... };
        
        HIP_CHECK(hipModuleLaunchKernel(
            kernel_,
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            shared_mem,
            nullptr,  // stream
            kernel_args,
            nullptr   // extra
        ));
        
        HIP_CHECK(hipDeviceSynchronize());
    }

private:
    hipModule_t module_;
    hipFunction_t kernel_;
};

//===----------------------------------------------------------------------===//
// LDS (Local Data Share) Memory Management
//===----------------------------------------------------------------------===//

class LdsAllocator {
public:
    // GFX942 has 64KB LDS per CU
    static constexpr size_t MAX_LDS_SIZE = 65536;
    
    static size_t getMaxLdsSize() { return MAX_LDS_SIZE; }
    
    static bool canAllocate(size_t size) {
        return size <= MAX_LDS_SIZE;
    }
    
    static void validateAllocation(size_t size) {
        if (!canAllocate(size)) {
            throw std::runtime_error(
                "LDS allocation of " + std::to_string(size) + 
                " bytes exceeds maximum of " + std::to_string(MAX_LDS_SIZE) + " bytes");
        }
    }
};

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

void* allocateDeviceMemory(size_t size) {
    void* ptr;
    HIP_CHECK(hipMalloc(&ptr, size));
    return ptr;
}

void freeDeviceMemory(void* ptr) {
    HIP_CHECK(hipFree(ptr));
}

void copyHostToDevice(void* dst, const void* src, size_t size) {
    HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
}

void copyDeviceToHost(void* dst, const void* src, size_t size) {
    HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));
}

void synchronize() {
    HIP_CHECK(hipDeviceSynchronize());
}

//===----------------------------------------------------------------------===//
// MFMA Information for GFX942
//===----------------------------------------------------------------------===//

struct MfmaShape {
    int m, n, k;
    std::string data_type;
    std::string instruction;
};

std::vector<MfmaShape> getSupportedMfmaShapes() {
    return {
        {32, 32, 8,  "f16",  "v_mfma_f32_32x32x8_f16"},
        {16, 16, 16, "f16",  "v_mfma_f32_16x16x16_f16"},
        {32, 32, 16, "bf16", "v_mfma_f32_32x32x16_bf16"},
        {16, 16, 4,  "f64",  "v_mfma_f64_16x16x4_f64"},
        {32, 32, 16, "i8",   "v_mfma_i32_32x32x16_i8"}
    };
}

void printMfmaInfo() {
    std::cout << "\nSupported MFMA instructions on GFX942:\n";
    for (const auto& shape : getSupportedMfmaShapes()) {
        std::cout << "  " << shape.instruction 
                  << " [" << shape.m << "x" << shape.n << "x" << shape.k << "]"
                  << " (" << shape.data_type << ")\n";
    }
}

} // namespace hip
} // namespace runtime
} // namespace cute

#endif // HAVE_HIP
