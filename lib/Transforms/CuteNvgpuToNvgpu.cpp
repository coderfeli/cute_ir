//===- CuteNvgpuToNvgpu.cpp - Lower cute_nvgpu to nvgpu dialect ----------===//
//
// This pass converts cute_nvgpu operations to MLIR nvgpu operations
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace cute {

//===----------------------------------------------------------------------===//
// Warpgroup MMA Lowering (SM90+)
//===----------------------------------------------------------------------===//

struct WarpgroupMmaOpLowering : public OpConversionPattern</* CuteNvgpuWarpgroupMmaOp */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteNvgpuWarpgroupMmaOp */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto fragA = adaptor.getFragA();
    auto fragB = adaptor.getFragB();
    auto fragC = adaptor.getFragC();
    auto atom = op.getAtom(); // MmaAtomType
    
    // Extract MMA shape from atom
    auto mmaShape = atom.getShape(); // [M, N, K]
    auto elemTypeA = atom.getElemTypeA();
    auto elemTypeB = atom.getElemTypeB();
    auto elemTypeC = atom.getElemTypeC();
    
    // Create nvgpu.warpgroup.mma.sync operation
    auto mmaOp = rewriter.create<nvgpu::WarpgroupMmaOp>(
        loc,
        fragC.getType(),    // Result type
        fragA,              // Operand A
        fragB,              // Operand B
        fragC               // Accumulator C
    );
    
    // Set MMA shape attribute
    mmaOp->setAttr("mmaShape", 
                   rewriter.getI64ArrayAttr({mmaShape[0], mmaShape[1], mmaShape[2]}));
    
    rewriter.replaceOp(op, mmaOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Warp MMA Lowering (SM80+)
//===----------------------------------------------------------------------===//

struct WarpMmaF16BF16OpLowering : public OpConversionPattern</* CuteNvgpuWarpMmaF16BF16Op */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteNvgpuWarpMmaF16BF16Op */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto atom = op.getAtom();
    auto mmaShape = atom.getShape(); // e.g., [16, 8, 16]
    
    // Create nvgpu.mma.sync operation
    auto mmaOp = rewriter.create<nvgpu::MmaSyncOp>(
        loc,
        adaptor.getFragC().getType(),
        adaptor.getFragA(),
        adaptor.getFragB(),
        adaptor.getFragC()
    );
    
    // Set shape and layout attributes
    mmaOp->setAttr("mmaShape", rewriter.getI64ArrayAttr(mmaShape));
    
    rewriter.replaceOp(op, mmaOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMA Load Lowering (SM90+)
//===----------------------------------------------------------------------===//

struct TmaLoadExecuteOpLowering : public OpConversionPattern</* CuteNvgpuTmaLoadExecuteOp */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteNvgpuTmaLoadExecuteOp */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto tmaDesc = adaptor.getTma(); // TmaLoadExecType
    auto srcGlobal = adaptor.getSrcGlobal();
    auto dstSmem = adaptor.getDstSmem();
    auto coords = adaptor.getCoords();
    
    // Extract TMA descriptor pointer from atom
    auto descPtr = rewriter.create</* AtomGetValueOp */>(
        loc, rewriter.getType<LLVM::LLVMPointerType>(),
        tmaDesc, rewriter.getStringAttr("tma_descriptor"));
    
    // Create nvgpu.tma.async.load
    auto tmaLoadOp = rewriter.create<nvgpu::TmaAsyncLoadOp>(
        loc,
        dstSmem,         // Destination shared memory
        descPtr,         // TMA descriptor
        coords,          // Tensor coordinates
        ValueRange{}     // Predicate (optional)
    );
    
    // Add barrier wait attribute
    auto barrier = tmaDesc.getBarrierType();
    tmaLoadOp->setAttr("barrier", barrier);
    
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Copy Operations Lowering
//===----------------------------------------------------------------------===//

struct LdmatrixOpLowering : public OpConversionPattern</* CuteNvgpuLdmatrixOp */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteNvgpuLdmatrixOp */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto srcSmem = adaptor.getSrcSmem();
    auto numRows = op.getNumRows();
    auto transpose = op.getTranspose();
    
    // Determine ldmatrix layout
    auto layout = transpose ? nvgpu::MMAMatrixLayoutAttr::get(
                                  rewriter.getContext(), 
                                  nvgpu::MMAMatrixLayout::ColMajor)
                            : nvgpu::MMAMatrixLayoutAttr::get(
                                  rewriter.getContext(),
                                  nvgpu::MMAMatrixLayout::RowMajor);
    
    // Create nvgpu.ldmatrix operation
    auto ldmatrixOp = rewriter.create<nvgpu::LdMatrixOp>(
        loc,
        op.getResult().getType(),
        srcSmem,
        /*indices=*/ValueRange{},
        transpose,
        numRows
    );
    
    rewriter.replaceOp(op, ldmatrixOp.getResult());
    return success();
  }
};

struct CpasyncG2SOpLowering : public OpConversionPattern</* CuteNvgpuCpasyncG2SOp */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteNvgpuCpasyncG2SOp */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto srcGlobal = adaptor.getSrcGlobal();
    auto dstSmem = adaptor.getDstSmem();
    auto bytes = op.getBytes();
    auto predicate = adaptor.getPredicate();
    
    // Create nvgpu.device_async_copy
    auto copyOp = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
        loc,
        dstSmem,              // Destination (shared memory)
        srcGlobal,            // Source (global memory)
        rewriter.getI32IntegerAttr(bytes),  // Copy size
        predicate.value_or(Value()),        // Predicate
        /*bypassL1=*/rewriter.getUnitAttr() // Cache hint
    );
    
    rewriter.eraseOp(op);
    return success();
  }
};

struct CpasyncCommitGroupOpLowering : public OpConversionPattern</* CuteNvgpuCpasyncCommitGroupOp */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteNvgpuCpasyncCommitGroupOp */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

struct CpasyncWaitGroupOpLowering : public OpConversionPattern</* CuteNvgpuCpasyncWaitGroupOp */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteNvgpuCpasyncWaitGroupOp */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto n = op.getN();
    rewriter.create<nvgpu::DeviceAsyncWaitOp>(
        op.getLoc(),
        rewriter.getI32IntegerAttr(n)
    );
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Barrier Operations Lowering (SM90+)
//===----------------------------------------------------------------------===//

struct MBarrierInitOpLowering : public OpConversionPattern</* CuteNvgpuMBarrierInitOp */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteNvgpuMBarrierInitOp */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto barrier = adaptor.getBarrier();
    auto count = op.getCount();
    
    // Create nvgpu.mbarrier.init
    rewriter.create<nvgpu::MBarrierInitOp>(
        loc,
        barrier,
        rewriter.getI32IntegerAttr(count)
    );
    
    rewriter.eraseOp(op);
    return success();
  }
};

struct MBarrierArriveOpLowering : public OpConversionPattern</* CuteNvgpuMBarrierArriveOp */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteNvgpuMBarrierArriveOp */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    rewriter.create<nvgpu::MBarrierArriveOp>(
        op.getLoc(),
        adaptor.getBarrier()
    );
    rewriter.eraseOp(op);
    return success();
  }
};

struct MBarrierWaitOpLowering : public OpConversionPattern</* CuteNvgpuMBarrierWaitOp */> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      /* CuteNvgpuMBarrierWaitOp */ op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto barrier = adaptor.getBarrier();
    auto phase = op.getPhase();
    
    rewriter.create<nvgpu::MBarrierWaitOp>(
        loc,
        barrier,
        rewriter.getI32IntegerAttr(phase)
    );
    
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct CuteNvgpuToNvgpuPass : public PassWrapper<CuteNvgpuToNvgpuPass,
                                                  OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CuteNvgpuToNvgpuPass)

  CuteNvgpuToNvgpuPass() = default;
  CuteNvgpuToNvgpuPass(const CuteNvgpuToNvgpuPass &) {}
  CuteNvgpuToNvgpuPass(std::string arch, bool tma) 
      : targetArch(arch), enableTMA(tma) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<nvgpu::NVGPUDialect,
                    gpu::GPUDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto gpuModule = getOperation();
    auto *context = &getContext();
    
    ConversionTarget target(*context);
    target.addLegalDialect<nvgpu::NVGPUDialect,
                          gpu::GPUDialect,
                          vector::VectorDialect>();
    target.addIllegalDialect</* CuteNvgpuDialect */>();
    
    RewritePatternSet patterns(context);
    
    // Add all lowering patterns
    patterns.add<WarpgroupMmaOpLowering,
                 WarpMmaF16BF16OpLowering,
                 LdmatrixOpLowering,
                 CpasyncG2SOpLowering,
                 CpasyncCommitGroupOpLowering,
                 CpasyncWaitGroupOpLowering>(context);
    
    // Add TMA patterns if enabled
    if (enableTMA) {
      patterns.add<TmaLoadExecuteOpLowering,
                   MBarrierInitOpLowering,
                   MBarrierArriveOpLowering,
                   MBarrierWaitOpLowering>(context);
    }
    
    if (failed(applyPartialConversion(gpuModule, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  Option<std::string> targetArch{
      *this, "target-arch",
      llvm::cl::desc("Target GPU architecture (sm_80, sm_90, sm_100)"),
      llvm::cl::init("sm_90")};
  
  Option<bool> enableTMA{
      *this, "enable-tma",
      llvm::cl::desc("Enable TMA operations (SM90+)"),
      llvm::cl::init(true)};
};

} // namespace cute

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createCuteNvgpuToNvgpuPass() {
  return std::make_unique<cute::CuteNvgpuToNvgpuPass>();
}

std::unique_ptr<Pass> createCuteNvgpuToNvgpuPass(std::string targetArch, bool enableTMA) {
  return std::make_unique<cute::CuteNvgpuToNvgpuPass>(targetArch, enableTMA);
}

} // namespace mlir
