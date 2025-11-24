// CuTe IR GEMM Example: C = A @ B (Matrix Multiply)
// Target: Hopper GPU (SM90) with Warpgroup MMA + TMA

module @cute_gemm {

  // Kernel function: GEMM using CuTe IR
  func.func @gemm_hopper(
    %A_ptr: !llvm.ptr,      // Global memory pointer for A (M x K)
    %B_ptr: !llvm.ptr,      // Global memory pointer for B (K x N)
    %C_ptr: !llvm.ptr,      // Global memory pointer for C (M x N)
    %M: index, %N: index, %K: index
  ) {
    
    //===--------------------------------------------------------------===//
    // Step 1: Define Layouts (rocdsl)
    //===--------------------------------------------------------------===//
    
    // Matrix A: (M, K) with row-major layout
    %shape_A = cute.make_shape %M, %K : !cute.shape<(?, ?)>
    %stride_A = cute.make_stride %K, %c1 : !cute.stride<(?, 1)>
    %layout_A = cute.make_layout %shape_A, %stride_A 
      : !cute.layout<(?, ?), (?, 1)>
    
    // Matrix B: (K, N) with column-major layout
    %shape_B = cute.make_shape %K, %N : !cute.shape<(?, ?)>
    %stride_B = cute.make_stride %c1, %K : !cute.stride<(1, ?)>
    %layout_B = cute.make_layout %shape_B, %stride_B
      : !cute.layout<(?, ?), (1, ?)>
    
    // Matrix C: (M, N) with row-major layout
    %shape_C = cute.make_shape %M, %N : !cute.shape<(?, ?)>
    %stride_C = cute.make_stride %N, %c1 : !cute.stride<(?, 1)>
    %layout_C = cute.make_layout %shape_C, %stride_C
      : !cute.layout<(?, ?), (?, 1)>
    
    //===--------------------------------------------------------------===//
    // Step 2: Create Tensors (rocdsl)
    //===--------------------------------------------------------------===//
    
    %tensor_A = cute.make_tensor %A_ptr, %layout_A 
      : !cute.tensor<f16, !cute.layout<...>>
    %tensor_B = cute.make_tensor %B_ptr, %layout_B
      : !cute.tensor<f16, !cute.layout<...>>
    %tensor_C = cute.make_tensor %C_ptr, %layout_C
      : !cute.tensor<f32, !cute.layout<...>>
    
    //===--------------------------------------------------------------===//
    // Step 3: Define MMA Atom (cute_nvgpu_ir)
    //===--------------------------------------------------------------===//
    
    // Warpgroup MMA: 64x128x16 (M x N x K) for FP16→FP32
    %mma_atom = cute_nvgpu.make_mma_atom !f16, !f16, !f32, [64, 128, 16], "SM90"
      : !cute_nvgpu.mma_atom<!f16, !f16, !f32, [64, 128, 16], SM90>
    
    // Thread value layout: 4 warps (128 threads per warpgroup)
    %shape_tv = cute.make_shape %c4, %c1 : !cute.shape<(4, 1)>
    %stride_tv = cute.make_stride %c1, %c0 : !cute.stride<(1, 0)>
    %layout_tv = cute.make_layout %shape_tv, %stride_tv
      : !cute.layout<(4,1), (1,0)>
    
    // Tiler: Hierarchical blocking
    %tile_m = cute.make_shape %c64 : !cute.shape<(64)>
    %tile_n = cute.make_shape %c128 : !cute.shape<(128)>
    %tile_k = cute.make_shape %c16 : !cute.shape<(16)>
    %tiler = cute.make_tile %tile_m, %tile_n, %tile_k
      : !cute.tile<(64, 128, 16)>
    
    // Construct TiledMma
    %tiled_mma = cute_nvgpu.make_tiled_mma %mma_atom, %layout_tv, %tiler
      : !cute_nvgpu.tiled_mma<...>
    
    //===--------------------------------------------------------------===//
    // Step 4: Define TMA Copy (cute_nvgpu_ir)
    //===--------------------------------------------------------------===//
    
    // TMA load descriptor for A (Global → Shared)
    %tma_load_A = cute_nvgpu.make_tma_load !f16, [64, 16], "128B"
      : !cute_nvgpu.tma_load<!f16, [64,16], 128B>
    
    // TMA load descriptor for B (Global → Shared)
    %tma_load_B = cute_nvgpu.make_tma_load !f16, [16, 128], "128B"
      : !cute_nvgpu.tma_load<!f16, [16,128], 128B>
    
    // Create barrier for TMA synchronization
    %barrier = nvgpu.mbarrier.create : !nvgpu.mbarrier
    cute_nvgpu.mbarrier_init %barrier, %c128 : (!nvgpu.mbarrier, i32)
    
    // Make executable TMA operations
    %exec_tma_A = cute_nvgpu.make_exec_tma_load %tma_load_A, %barrier
      : !cute_nvgpu.tma_load_exec<...>
    %exec_tma_B = cute_nvgpu.make_exec_tma_load %tma_load_B, %barrier
      : !cute_nvgpu.tma_load_exec<...>
    
    //===--------------------------------------------------------------===//
    // Step 5: Thread Partitioning (rocdsl)
    //===--------------------------------------------------------------===//
    
    %thread_idx = gpu.thread_id : index
    %block_idx_m = gpu.block_id x : index
    %block_idx_n = gpu.block_id y : index
    
    // Partition tensors across threads
    %thr_A = cute.local_partition %tensor_A, %tiler, %thread_idx
      : !cute.tensor<f16, ...>
    %thr_B = cute.local_partition %tensor_B, %tiler, %thread_idx
      : !cute.tensor<f16, ...>
    %thr_C = cute.local_partition %tensor_C, %tiler, %thread_idx
      : !cute.tensor<f32, ...>
    
    //===--------------------------------------------------------------===//
    // Step 6: K-loop with TMA + Warpgroup MMA
    //===--------------------------------------------------------------===//
    
    %c0 = arith.constant 0 : index
    %K_tiles = arith.divui %K, %c16 : index
    
    scf.for %k_tile = %c0 to %K_tiles step %c1 {
      
      // === TMA Load (Global → Shared) ===
      
      // Load A tile: A[block_m * 64 : (block_m+1) * 64, k_tile * 16 : (k_tile+1) * 16]
      %coord_A = cute.make_coord %block_idx_m, %k_tile : !cute.coord<(?, ?)>
      %smem_A = memref.alloc() : memref<64x16xf16, #gpu.address_space<workgroup>>
      
      cute_nvgpu.mbarrier_arrive %barrier
      cute_nvgpu.tma_load_execute %exec_tma_A, %A_ptr, %smem_A, %block_idx_m, %k_tile
      
      // Load B tile: B[k_tile * 16 : (k_tile+1) * 16, block_n * 128 : (block_n+1) * 128]
      %coord_B = cute.make_coord %k_tile, %block_idx_n : !cute.coord<(?, ?)>
      %smem_B = memref.alloc() : memref<16x128xf16, #gpu.address_space<workgroup>>
      
      cute_nvgpu.tma_load_execute %exec_tma_B, %B_ptr, %smem_B, %k_tile, %block_idx_n
      
      // Wait for TMA completion
      %phase = arith.constant 0 : i32
      cute_nvgpu.mbarrier_wait %barrier, %phase
      
      // === Warpgroup MMA ===
      
      // Make fragments from shared memory
      %layout_smem_A = cute.make_layout ...
      %tensor_smem_A = cute.make_tensor %smem_A, %layout_smem_A
      %fragA = cute.mma_make_fragment_a %tiled_mma, %tensor_smem_A
        : !cute.tensor<f16, ...>
      
      %layout_smem_B = cute.make_layout ...
      %tensor_smem_B = cute.make_tensor %smem_B, %layout_smem_B
      %fragB = cute.mma_make_fragment_b %tiled_mma, %tensor_smem_B
        : !cute.tensor<f16, ...>
      
      %fragC_in = cute.mma_make_fragment_c %tiled_mma, %thr_C
        : !cute.tensor<f32, ...>
      
      // Execute warpgroup MMA
      %fragC_out = cute_nvgpu.warpgroup_mma %fragA, %fragB, %fragC_in, %mma_atom
        : (!cute.tensor<f16, ...>, !cute.tensor<f16, ...>, 
           !cute.tensor<f32, ...>, !cute_nvgpu.mma_atom<...>) 
        -> !cute.tensor<f32, ...>
      
      // Update C accumulator
      cute.copy %fragC_out, %thr_C
      
      memref.dealloc %smem_A : memref<64x16xf16, #gpu.address_space<workgroup>>
      memref.dealloc %smem_B : memref<16x128xf16, #gpu.address_space<workgroup>>
    }
    
    //===--------------------------------------------------------------===//
    // Step 7: Store Results (Shared → Global)
    //===--------------------------------------------------------------===//
    
    // Copy accumulated results back to global memory
    cute.copy %thr_C, %tensor_C
    
    return
  }
  
  //===------------------------------------------------------------------===//
  // Host-Side Launcher (Simplified)
  //===------------------------------------------------------------------===//
  
  func.func @main() {
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    %M = %c1024
    %N = %c2048
    %K = %c1024
    
    // Allocate host memory
    %A = memref.alloc(%M, %K) : memref<?x?xf16>
    %B = memref.alloc(%K, %N) : memref<?x?xf16>
    %C = memref.alloc(%M, %N) : memref<?x?xf32>
    
    // Copy to device (simplified)
    %A_gpu = gpu.alloc(%M, %K) : memref<?x?xf16>
    %B_gpu = gpu.alloc(%K, %N) : memref<?x?xf16>
    %C_gpu = gpu.alloc(%M, %N) : memref<?x?xf32>
    
    gpu.memcpy %A_gpu, %A : memref<?x?xf16>, memref<?x?xf16>
    gpu.memcpy %B_gpu, %B : memref<?x?xf16>, memref<?x?xf16>
    
    // Launch kernel
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %grid_m = arith.divui %M, %c64 : index  // 1024 / 64 = 16 blocks
    %grid_n = arith.divui %N, %c128 : index // 2048 / 128 = 16 blocks
    
    gpu.launch_func @gemm_hopper blocks in (%grid_m, %grid_n, %c1) 
                                  threads in (%c128, %c1, %c1)
                                  args(%A_gpu, %B_gpu, %C_gpu, %M, %N, %K)
    
    // Copy results back
    gpu.memcpy %C, %C_gpu : memref<?x?xf32>, memref<?x?xf32>
    
    return
  }
}
