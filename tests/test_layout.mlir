// RUN: mlir-opt %s -pass-pipeline='builtin.module(cute-canonicalize)' | FileCheck %s

// Test basic layout creation and manipulation

module {
  func.func @test_make_layout() -> !cute.layout<2> {
    // Create shape and stride
    %shape = cute.make_shape 8, 16 : !cute.shape<2>
    %stride = cute.make_stride 1, 8 : !cute.stride<2>
    
    // Create layout from shape and stride
    %layout = cute.make_layout %shape, %stride : !cute.layout<2>
    
    // CHECK: cute.make_layout
    return %layout : !cute.layout<2>
  }
  
  func.func @test_layout_size(%arg0: !cute.layout<2>) -> !cute.int {
    // Query layout size
    %size = cute.size %arg0 : !cute.layout<2> -> !cute.int
    
    // CHECK: cute.size
    return %size : !cute.int
  }
  
  func.func @test_coord_to_index() {
    %layout = cute.make_layout (!cute.shape<2>), (!cute.stride<2>) : !cute.layout<2>
    %coord = cute.make_coord 3, 5 : !cute.coord<2>
    
    // Convert coordinate to linear index
    %idx = cute.crd2idx %layout, %coord : !cute.layout<2>, !cute.coord<2> -> !cute.int
    
    // CHECK: cute.crd2idx
    // Expected: 3 + 5*8 = 43
    return
  }
}
