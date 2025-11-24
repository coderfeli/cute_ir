// Test cute.cosize operation
func.func @test_cosize() -> index {
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  
  %shape = cute.make_shape %c8, %c128 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %cosize = cute.cosize %layout : !cute.layout<2> -> index
  
  return %cosize : index
}
