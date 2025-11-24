// Test partial pass lowering (crd2idx only)
module {
  func.func @test_crd2idx_simple(%c: !cute.coord<2>, %l: !cute.layout<2>) {
    %idx = cute.crd2idx %c, %l : (!cute.coord<2>, !cute.layout<2>) -> !cute.int
    // Note: %idx is lowered to index type but not used, so no type conversion error
    return
  }
}
