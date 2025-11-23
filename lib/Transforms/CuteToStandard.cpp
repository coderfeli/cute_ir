//===- CuteToStandard.cpp - Lower CuTe IR to Standard Dialects -----------===//
//
// This pass converts cute_ir operations to standard MLIR dialects
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace cute {

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert cute.crd2idx to arithmetic operations
/// Formula: idx = coord[0]*stride[0] + coord[1]*stride[1] + ...
struct Crd2IdxOpLowering : public OpConversionPattern</* CuteCrd2IdxOp */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteCrd2IdxOp */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto coordType = op.getCoord().getType(); // CoordType
    auto layoutType = op.getLayout().getType(); // LayoutType
    
    // Extract shape and stride from layout
    auto shape = layoutType.getShape(); // ArrayRef<int64_t>
    auto stride = layoutType.getStride(); // ArrayRef<int64_t>
    int64_t rank = shape.size();
    
    // Build: result = coord[0] * stride[0] + coord[1] * stride[1] + ...
    Value result = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    
    for (int64_t i = 0; i < rank; ++i) {
      // Extract coord[i]
      Value coordI = rewriter.create</* ExtractCoordOp */>(loc, adaptor.getCoord(), i);
      
      // Multiply by stride[i]
      Value strideI = rewriter.create<arith::ConstantIndexOp>(loc, stride[i]);
      Value term = rewriter.create<arith::MulIOp>(loc, coordI, strideI);
      
      // Accumulate
      result = rewriter.create<arith::AddIOp>(loc, result, term);
    }
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert cute.copy to memref load/store loop
/// Generates vectorized loads when layout permits
struct CopyOpLowering : public OpConversionPattern</* CuteCopyOp */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteCopyOp */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto src = adaptor.getSrc(); // TensorType
    auto dst = adaptor.getDst(); // TensorType
    
    auto srcLayout = src.getType().getLayout();
    auto dstLayout = dst.getType().getLayout();
    
    // Get iteration bounds from layout
    auto shape = srcLayout.getShape();
    int64_t size = 1;
    for (auto dim : shape)
      size *= dim;
    
    // Detect vectorization opportunities
    int64_t vectorWidth = analyzeVectorWidth(srcLayout, dstLayout);
    
    if (vectorWidth > 1) {
      // Vectorized copy
      generateVectorizedCopy(rewriter, loc, src, dst, size, vectorWidth);
    } else {
      // Scalar copy loop
      auto lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto ub = rewriter.create<arith::ConstantIndexOp>(loc, size);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      
      auto loop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
      rewriter.setInsertionPointToStart(loop.getBody());
      
      // Inside loop: load from src, store to dst
      Value idx = loop.getInductionVar();
      Value data = rewriter.create<memref::LoadOp>(loc, src, idx);
      rewriter.create<memref::StoreOp>(loc, data, dst, idx);
      
      rewriter.setInsertionPointAfter(loop);
    }
    
    rewriter.eraseOp(op);
    return success();
  }

private:
  // Analyze layout to determine vectorization width
  int64_t analyzeVectorWidth(LayoutType srcLayout, LayoutType dstLayout) const {
    // Check if innermost dimension is contiguous (stride == 1)
    auto srcStride = srcLayout.getStride();
    auto dstStride = dstLayout.getStride();
    
    if (srcStride.back() == 1 && dstStride.back() == 1) {
      // Contiguous access - can vectorize
      int64_t innerDim = srcLayout.getShape().back();
      
      // Common vector widths: 1, 2, 4, 8, 16
      for (int64_t width : {16, 8, 4, 2}) {
        if (innerDim % width == 0)
          return width;
      }
    }
    
    return 1; // No vectorization
  }
  
  void generateVectorizedCopy(ConversionPatternRewriter &rewriter,
                               Location loc, Value src, Value dst,
                               int64_t size, int64_t vectorWidth) const {
    // Generate: for (i = 0; i < size; i += vectorWidth)
    //             vec = vector.load src[i]
    //             vector.store vec, dst[i]
    
    auto elemType = src.getType().getElementType();
    auto vecType = VectorType::get({vectorWidth}, elemType);
    
    auto lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto ub = rewriter.create<arith::ConstantIndexOp>(loc, size);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
    
    auto loop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
    rewriter.setInsertionPointToStart(loop.getBody());
    
    Value idx = loop.getInductionVar();
    Value vec = rewriter.create<vector::LoadOp>(loc, vecType, src, ValueRange{idx});
    rewriter.create<vector::StoreOp>(loc, vec, dst, ValueRange{idx});
  }
};

/// Convert cute.local_partition to thread-indexed access
struct LocalPartitionOpLowering : public OpConversionPattern</* CuteLocalPartitionOp */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteLocalPartitionOp */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto tensor = adaptor.getTensor();
    auto tiler = adaptor.getTiler(); // TileType
    auto threadIdx = adaptor.getThreadIdx();
    
    // Compute thread-local base offset
    // offset = threadIdx * tile_size
    auto tileLayout = tiler.getType().getLayouts()[0];
    int64_t tileSize = computeTileSize(tileLayout);
    
    Value tileSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, tileSize);
    Value baseOffset = rewriter.create<arith::MulIOp>(loc, threadIdx, tileSizeVal);
    
    // Create new tensor view with offset
    auto newTensor = rewriter.create</* MakeTensorOp */>(
        loc, tensor, baseOffset, tiler);
    
    rewriter.replaceOp(op, newTensor);
    return success();
  }

private:
  int64_t computeTileSize(LayoutType layout) const {
    int64_t size = 1;
    for (auto dim : layout.getShape())
      size *= dim;
    return size;
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct CuteToStandardPass : public PassWrapper<CuteToStandardPass, 
                                                OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CuteToStandardPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, 
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();
    
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, 
                          scf::SCFDialect,
                          memref::MemRefDialect,
                          func::FuncDialect>();
    target.addIllegalDialect</* CuteDialect */>();
    
    RewritePatternSet patterns(context);
    patterns.add<Crd2IdxOpLowering,
                 CopyOpLowering,
                 LocalPartitionOpLowering>(context);
    
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace cute

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createCuteToStandardPass() {
  return std::make_unique<cute::CuteToStandardPass>();
}

} // namespace mlir
