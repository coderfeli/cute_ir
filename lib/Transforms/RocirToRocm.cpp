//===- RocirToRocm.cpp - Lower Rocir IR to ROCm Dialect ------------------===//
//
// This pass converts rocdsl operations to the ROCm dialect for GFX942
//
//===----------------------------------------------------------------------===//

#include "rocir/RocirDialect.h"
#include "rocir/RocirOps.h"
#include "rocir/RocirRocmDialect.h"
#include "rocir/RocirRocmOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace rocir {

//===----------------------------------------------------------------------===//
// Type Conversions
//===----------------------------------------------------------------------===//

namespace {

/// Type converter for Rocir to ROCm lowering
class RocirToRocmTypeConverter : public TypeConverter {
public:
  RocirToRocmTypeConverter() {
    // Keep MLIR builtin types (index, integers, floats) as-is
    addConversion([](Type type) { return type; });
    
    // TODO: Add conversions for Rocir types to ROCm types
    // For now, most types remain unchanged as they're already in the correct dialect
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Lower rocir.product_each to a sequence of shape manipulations
struct ProductEachOpLowering : public OpConversionPattern<ProductEachOp> {
  using OpConversionPattern::OpConversionPattern>
  
  LogicalResult matchAndRewrite(ProductEachOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto shape = adaptor.getShape();
    
    // For now, product_each is a metadata operation that gets constant-folded
    // during canonicalization. In a full implementation, we would:
    // 1. Extract the shape structure
    // 2. Compute products at each hierarchical level
    // 3. Reconstruct a new shape with computed products
    
    // Keep the operation as-is for now - it will be lowered in a later pass
    // or constant-folded if the shape is static
    rewriter.replaceOp(op, shape);
    return success();
  }
};

/// Lower rocir.make_layout_tv to rocir_rocm operations
struct MakeLayoutTVOpLowering : public OpConversionPattern<MakeLayoutTVOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(MakeLayoutTVOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // Algorithm:
    // 1. layout_mn = raked_product(thr_layout, val_layout)
    auto layoutMN = rewriter.create<RakedProductOp>(
        loc, adaptor.getThrLayout(), adaptor.getValLayout());
    
    // 2. Get shape from layout_mn
    auto shapeType = ShapeType::get(rewriter.getContext(), 2);
    auto shape = rewriter.create<GetShapeOp>(loc, shapeType, layoutMN.getResult());
    
    // 3. tiler_mn = product_each(shape)
    auto tilerMN = rewriter.create<ProductEachOp>(loc, shapeType, shape.getResult());
    
    // 4. Compute layout_tv
    // Full implementation would compute:
    // layout_tv = composition(right_inverse(layout_mn), make_layout((thr_size, val_size)))
    
    // For now, use a simplified approach:
    // Get thr_size and val_size
    auto thrSize = rewriter.create<SizeOp>(
        loc, rewriter.getIndexType(), adaptor.getThrLayout());
    auto valSize = rewriter.create<SizeOp>(
        loc, rewriter.getIndexType(), adaptor.getValLayout());
    
    // Create a shape for (thr_size, val_size)
    auto tvShapeType = ShapeType::get(rewriter.getContext(), 2);
    SmallVector<Value> tvShapeValues = {thrSize.getResult(), valSize.getResult()};
    auto tvShape = rewriter.create<MakeShapeOp>(loc, tvShapeType, tvShapeValues);
    
    // Compute right inverse of layout_mn
    auto layoutType = LayoutType::get(rewriter.getContext(), 2);
    auto layoutMNInv = rewriter.create<RightInverseOp>(
        loc, layoutType, layoutMN.getResult());
    
    // Create a layout from tv_shape
    // For simplicity, use a default stride (column-major)
    auto strideType = StrideType::get(rewriter.getContext(), 2);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> strideValues = {one.getResult(), thrSize.getResult()};
    auto tvStride = rewriter.create<MakeStrideOp>(loc, strideType, strideValues);
    auto tvLayout = rewriter.create<MakeLayoutOp>(
        loc, layoutType, tvShape.getResult(), tvStride.getResult());
    
    // Compose to get final layout_tv
    auto layoutTV = rewriter.create<CompositionOp>(
        loc, layoutType, layoutMNInv.getResult(), tvLayout.getResult());
    
    rewriter.replaceOp(op, {tilerMN.getResult(), layoutTV.getResult()});
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// RocirToRocm Pass
//===----------------------------------------------------------------------===//

namespace {

/// Pass to lower Rocir IR to the ROCm dialect for AMD GFX942.
struct RocirToRocmPass : public PassWrapper<RocirToRocmPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocirToRocmPass)
  
  StringRef getArgument() const final { return "rocir-to-rocm"; }
  StringRef getDescription() const final {
    return "Lower Rocir IR to ROCm dialect for AMD GFX942";
  }
  
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();
    
    // Mark as targeting AMD GFX942
    module->setAttr("rocir.target_arch", 
                    StringAttr::get(context, "gfx942"));
    module->setAttr("rocir.target_vendor",
                    StringAttr::get(context, "amd"));
    
    // Set up type converter
    RocirToRocmTypeConverter typeConverter;
    
    // Set up conversion target
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, 
                           func::FuncDialect,
                           gpu::GPUDialect,
                           memref::MemRefDialect,
                           rocir_rocm::RocirRocmDialect>();
    
    // Mark specific Rocir ops as illegal (to be lowered)
    target.addIllegalOp<MakeLayoutTVOp>();
    
    // Most Rocir ops remain legal (they're lowered in later passes)
    target.addLegalDialect<RocirDialect>();
    
    // Populate patterns
    RewritePatternSet patterns(context);
    patterns.add<MakeLayoutTVOpLowering>(typeConverter, context);
    
    // Apply conversion (partial - not all ops need lowering at this stage)
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createRocirToRocmPass() {
  return std::make_unique<RocirToRocmPass>();
}

} // namespace rocir
} // namespace mlir
