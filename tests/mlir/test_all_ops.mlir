// RUN: cute-opt %s --cute-to-standard | FileCheck %s

// Comprehensive test covering all layout operations and lowering

module {
  // ============================================================================
  // Basic Operations Tests
  // ============================================================================
  
  // CHECK-LABEL: @test_make_shape
  func.func @test_make_shape() -> index {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    
    // CHECK: cute.make_shape
    %shape = cute.make_shape %c8, %c16 : (index, index) -> !cute.shape<2>
    
    // CHECK: arith.muli
    // CHECK: arith.muli
    %size = cute.size %shape : !cute.shape<2> -> index
    
    // CHECK: return
    return %size : index
  }
  
  // CHECK-LABEL: @test_make_layout
  func.func @test_make_layout() -> index {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    // CHECK: cute.make_shape
    %shape = cute.make_shape %c8, %c16 : (index, index) -> !cute.shape<2>
    // CHECK: cute.make_stride
    %stride = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
    
    // CHECK: cute.make_layout
    %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: arith.muli
    %size = cute.size %layout : !cute.layout<2> -> index
    
    return %size : index
  }
  
  // CHECK-LABEL: @test_get_shape_stride
  func.func @test_get_shape_stride() -> index {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    
    %shape = cute.make_shape %c4, %c8 : (index, index) -> !cute.shape<2>
    %stride = cute.make_stride %c1, %c4 : (index, index) -> !cute.stride<2>
    %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.get_shape
    %extracted_shape = cute.get_shape %layout : !cute.layout<2> -> !cute.shape<2>
    // CHECK: cute.get_stride
    %extracted_stride = cute.get_stride %layout : !cute.layout<2> -> !cute.stride<2>
    
    %size1 = cute.size %extracted_shape : !cute.shape<2> -> index
    %size2 = cute.cosize %layout : !cute.layout<2> -> index
    
    %result = arith.addi %size1, %size2 : index
    return %result : index
  }
  
  // ============================================================================
  // Product Operations Tests (Tiling)
  // ============================================================================
  
  // CHECK-LABEL: @test_logical_product
  func.func @test_logical_product() -> index {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    
    // Base layout: 16x32 column-major
    %base_shape = cute.make_shape %c16, %c32 : (index, index) -> !cute.shape<2>
    %base_stride = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
    %base = cute.make_layout %base_shape, %base_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // Tiler: 4x8
    %tile_shape = cute.make_shape %c4, %c8 : (index, index) -> !cute.shape<2>
    %tile_stride = cute.make_stride %c1, %c4 : (index, index) -> !cute.stride<2>
    %tiler = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.logical_product
    // CHECK: cute.composition
    // CHECK: arith.muli
    %tiled = cute.logical_product %base, %tiler : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<4>
    
    %size = cute.size %tiled : !cute.layout<4> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_zipped_product
  func.func @test_zipped_product() -> index {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = cute.make_shape %c8, %c16 : (index, index) -> !cute.shape<2>
    %base_stride = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
    %base = cute.make_layout %base_shape, %base_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    %tile_shape = cute.make_shape %c2, %c4 : (index, index) -> !cute.shape<2>
    %tile_stride = cute.make_stride %c1, %c2 : (index, index) -> !cute.stride<2>
    %tiler = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.zipped_product
    // CHECK: cute.logical_product
    %zipped = cute.zipped_product %base, %tiler : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<4>
    
    %size = cute.size %zipped : !cute.layout<4> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_tiled_product
  func.func @test_tiled_product() -> index {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = cute.make_shape %c32, %c64 : (index, index) -> !cute.shape<2>
    %base_stride = cute.make_stride %c1, %c32 : (index, index) -> !cute.stride<2>
    %base = cute.make_layout %base_shape, %base_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    %tile_shape = cute.make_shape %c8, %c16 : (index, index) -> !cute.shape<2>
    %tile_stride = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
    %tiler = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.tiled_product
    // CHECK: cute.logical_product
    %tiled = cute.tiled_product %base, %tiler : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<4>
    
    %size = cute.size %tiled : !cute.layout<4> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_flat_product
  func.func @test_flat_product() -> index {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = cute.make_shape %c16, %c8 : (index, index) -> !cute.shape<2>
    %base_stride = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
    %base = cute.make_layout %base_shape, %base_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    %tile_shape = cute.make_shape %c4, %c2 : (index, index) -> !cute.shape<2>
    %tile_stride = cute.make_stride %c1, %c4 : (index, index) -> !cute.stride<2>
    %tiler = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.flat_product
    // CHECK: cute.logical_product
    %flat = cute.flat_product %base, %tiler : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<2>
    
    %size = cute.size %flat : !cute.layout<2> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_raked_product
  func.func @test_raked_product() -> index {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = cute.make_shape %c32, %c8 : (index, index) -> !cute.shape<2>
    %base_stride = cute.make_stride %c1, %c32 : (index, index) -> !cute.stride<2>
    %base = cute.make_layout %base_shape, %base_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    %tile_shape = cute.make_shape %c8, %c4 : (index, index) -> !cute.shape<2>
    %tile_stride = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
    %tiler = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.raked_product
    // CHECK: cute.logical_product
    %raked = cute.raked_product %base, %tiler : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<4>
    
    %size = cute.size %raked : !cute.layout<4> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_blocked_product
  func.func @test_blocked_product() -> index {
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    
    %base_shape = cute.make_shape %c64, %c16 : (index, index) -> !cute.shape<2>
    %base_stride = cute.make_stride %c1, %c64 : (index, index) -> !cute.stride<2>
    %base = cute.make_layout %base_shape, %base_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    %tile_shape = cute.make_shape %c8, %c4 : (index, index) -> !cute.shape<2>
    %tile_stride = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
    %tiler = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.blocked_product
    // CHECK: cute.logical_product
    %blocked = cute.blocked_product %base, %tiler : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<4>
    
    %size = cute.size %blocked : !cute.layout<4> -> index
    return %size : index
  }
  
  // ============================================================================
  // Divide Operations Tests (Partitioning)
  // ============================================================================
  
  // CHECK-LABEL: @test_logical_divide
  func.func @test_logical_divide() -> index {
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    
    // Global layout: 128x256
    %global_shape = cute.make_shape %c128, %c256 : (index, index) -> !cute.shape<2>
    %global_stride = cute.make_stride %c1, %c128 : (index, index) -> !cute.stride<2>
    %global = cute.make_layout %global_shape, %global_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // Tile: 16x32
    %tile_shape = cute.make_shape %c16, %c32 : (index, index) -> !cute.shape<2>
    %tile_stride = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
    %tile = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.logical_divide
    // CHECK: cute.composition
    %partitioned = cute.logical_divide %global, %tile : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<4>
    
    %size = cute.size %partitioned : !cute.layout<4> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_zipped_divide
  func.func @test_zipped_divide() -> index {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    %global_shape = cute.make_shape %c64, %c128 : (index, index) -> !cute.shape<2>
    %global_stride = cute.make_stride %c1, %c64 : (index, index) -> !cute.stride<2>
    %global = cute.make_layout %global_shape, %global_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    %tile_shape = cute.make_shape %c8, %c16 : (index, index) -> !cute.shape<2>
    %tile_stride = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
    %tile = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.zipped_divide
    // CHECK: cute.logical_divide
    %zipped = cute.zipped_divide %global, %tile : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<4>
    
    %size = cute.size %zipped : !cute.layout<4> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_tiled_divide
  func.func @test_tiled_divide() -> index {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    
    %global_shape = cute.make_shape %c32, %c64 : (index, index) -> !cute.shape<2>
    %global_stride = cute.make_stride %c1, %c32 : (index, index) -> !cute.stride<2>
    %global = cute.make_layout %global_shape, %global_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    %tile_shape = cute.make_shape %c4, %c8 : (index, index) -> !cute.shape<2>
    %tile_stride = cute.make_stride %c1, %c4 : (index, index) -> !cute.stride<2>
    %tile = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.tiled_divide
    // CHECK: cute.logical_divide
    %tiled = cute.tiled_divide %global, %tile : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<4>
    
    %size = cute.size %tiled : !cute.layout<4> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_flat_divide
  func.func @test_flat_divide() -> index {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    
    %global_shape = cute.make_shape %c16, %c32 : (index, index) -> !cute.shape<2>
    %global_stride = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
    %global = cute.make_layout %global_shape, %global_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    %tile_shape = cute.make_shape %c4, %c8 : (index, index) -> !cute.shape<2>
    %tile_stride = cute.make_stride %c1, %c4 : (index, index) -> !cute.stride<2>
    %tile = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.flat_divide
    // CHECK: cute.logical_divide
    %flat = cute.flat_divide %global, %tile : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<2>
    
    %size = cute.size %flat : !cute.layout<2> -> index
    return %size : index
  }
  
  // ============================================================================
  // Local Operations Tests (Thread/Block Partitioning)
  // ============================================================================
  
  // CHECK-LABEL: @test_local_partition
  func.func @test_local_partition() -> index {
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    // Global tensor: 128x256
    %global_shape = cute.make_shape %c128, %c256 : (index, index) -> !cute.shape<2>
    %global_stride = cute.make_stride %c1, %c128 : (index, index) -> !cute.stride<2>
    %global = cute.make_layout %global_shape, %global_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // Thread tile: 8x16
    %tile_shape = cute.make_shape %c8, %c16 : (index, index) -> !cute.shape<2>
    %tile_stride = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
    %tile = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.local_partition
    // CHECK: cute.logical_divide
    %thread_data = cute.local_partition %global, %tile, %c0 : (!cute.layout<2>, !cute.layout<2>, index) -> !cute.layout<2>
    
    %size = cute.size %thread_data : !cute.layout<2> -> index
    return %size : index
  }
  
  // CHECK-LABEL: @test_local_tile
  func.func @test_local_tile() -> index {
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    // Global tensor: 128x256
    %global_shape = cute.make_shape %c128, %c256 : (index, index) -> !cute.shape<2>
    %global_stride = cute.make_stride %c1, %c128 : (index, index) -> !cute.stride<2>
    %global = cute.make_layout %global_shape, %global_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CTA tile shape: 32x64
    %cta_shape = cute.make_shape %c32, %c64 : (index, index) -> !cute.shape<2>
    
    // CTA coordinates: (0, 0)
    %cta_coord = cute.make_shape %c0, %c0 : (index, index) -> !cute.shape<2>
    
    // CHECK: cute.local_tile
    // CHECK: cute.logical_divide
    %cta_tile = cute.local_tile %global, %cta_shape, %cta_coord : (!cute.layout<2>, !cute.shape<2>, !cute.shape<2>) -> !cute.layout<2>
    
    %size = cute.size %cta_tile : !cute.layout<2> -> index
    return %size : index
  }
  
  // ============================================================================
  // Composition Test
  // ============================================================================
  
  // CHECK-LABEL: @test_composition
  func.func @test_composition() -> index {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    
    %shape_a = cute.make_shape %c8, %c16 : (index, index) -> !cute.shape<2>
    %stride_a = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
    %layout_a = cute.make_layout %shape_a, %stride_a : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    %shape_b = cute.make_shape %c4, %c2 : (index, index) -> !cute.shape<2>
    %stride_b = cute.make_stride %c2, %c1 : (index, index) -> !cute.stride<2>
    %layout_b = cute.make_layout %shape_b, %stride_b : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // CHECK: cute.composition
    // CHECK: arith.muli
    // CHECK: arith.addi
    %composed = cute.composition %layout_a, %layout_b : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<2>
    
    %size = cute.size %composed : !cute.layout<2> -> index
    return %size : index
  }
}
