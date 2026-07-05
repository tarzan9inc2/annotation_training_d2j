#!/usr/bin/env python3
"""
PyTorchモデルをONNX経由でTensorRTに変換するスクリプト
"""
import torch
import sys
import os
import tensorrt as trt
import numpy as np

# model_catalogをインポート
sys.path.append('/home/jetson/mycar')
from model_catalog import get_model, load_model_weights

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def convert_pytorch_to_onnx(model_path, onnx_path, model_type='edgenext_xx_small', input_size=(224, 224)):
    """PyTorchモデルをONNXに変換"""
    print(f"Step 1: Converting {model_path} to ONNX...")
    print(f"Model type: {model_type}")
    print(f"Input size: {input_size}")

    # デバイス設定
    device = torch.device('cuda')

    # モデル読み込み
    print("Loading PyTorch model...")
    model = get_model(model_type, pretrained=False, input_size=input_size)
    model = load_model_weights(model, model_path, device)
    model = model.to(device)
    model.eval()
    print("PyTorch model loaded successfully")

    # ダミー入力作成
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).cuda()

    # ONNX変換
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"ONNX model saved to: {onnx_path}")

    # ONNX検証
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation: PASSED")

    return onnx_path

def build_engine_from_onnx(onnx_path, engine_path, fp16_mode=True):
    """ONNXからTensorRTエンジンをビルド"""
    print(f"\nStep 2: Building TensorRT engine from {onnx_path}...")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # ONNX読み込み
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # ビルダー設定
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")

    # プロファイル設定（dynamic shapes用）
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
    config.add_optimization_profile(profile)

    # エンジンビルド
    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None

    # エンジン保存
    print(f"Saving TensorRT engine to: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print("TensorRT engine build completed!")
    return engine_path

def benchmark_model(model_path, engine_path, num_iterations=100):
    """PyTorchとTensorRTの推論速度を比較"""
    print(f"\nStep 3: Benchmarking...")

    # PyTorchベンチマーク
    device = torch.device('cuda')
    model = get_model('edgenext_xx_small', pretrained=False)
    model = load_model_weights(model, model_path, device)
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).cuda()

    # ウォームアップ
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # PyTorch速度測定
    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iterations * 1000

    # TensorRTベンチマーク
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # 入出力バッファ確保
    import pycuda.driver as cuda
    import pycuda.autoinit

    # エンジンの入出力情報を確認
    print(f"\nTensorRT Engine Info:")
    print(f"  Inputs: {[engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]}")
    print(f"  Outputs: {[engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]}")

    # 入力バッファ
    h_input = cuda.pagelocked_empty(trt.volume((1, 3, 224, 224)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    context.set_tensor_address('input', int(d_input))

    # 出力バッファ (outputの形状を確認)
    output_shape = context.get_tensor_shape('output')
    print(f"  Output shape: {output_shape}")
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_output = cuda.mem_alloc(h_output.nbytes)
    context.set_tensor_address('output', int(d_output))

    stream = cuda.Stream()

    # TensorRT 10.x用の入力形状設定
    context.set_input_shape('input', (1, 3, 224, 224))

    # ウォームアップ
    for _ in range(10):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # TensorRT速度測定
    start = time.time()
    for _ in range(num_iterations):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    trt_time = (time.time() - start) / num_iterations * 1000

    # 結果表示
    print(f"\n{'='*60}")
    print(f"Benchmark Results ({num_iterations} iterations)")
    print(f"{'='*60}")
    print(f"PyTorch inference time: {pytorch_time:.2f} ms/frame ({1000/pytorch_time:.1f} fps)")
    print(f"TensorRT inference time: {trt_time:.2f} ms/frame ({1000/trt_time:.1f} fps)")
    print(f"Speedup: {pytorch_time/trt_time:.2f}x")
    print(f"{'='*60}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert PyTorch model to TensorRT via ONNX')
    parser.add_argument('model_path', type=str, help='Path to PyTorch model (.pth)')
    parser.add_argument('--onnx', type=str, default=None, help='Output path for ONNX model')
    parser.add_argument('--engine', type=str, default=None, help='Output path for TensorRT engine')
    parser.add_argument('--model_type', type=str, default='edgenext_xx_small', help='Model architecture type')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16 mode')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark comparison')

    args = parser.parse_args()

    # モデルファイルの存在チェック
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        print(f"\nSearching for similar files in models directory...")

        # modelsディレクトリから類似ファイルを検索
        model_dir = os.path.dirname(args.model_path) or 'models'
        model_name = os.path.basename(args.model_path)

        if os.path.exists(model_dir):
            # .pthファイルを検索
            pth_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

            if pth_files:
                print(f"\nAvailable PyTorch models in {model_dir}:")
                # 日付順にソート（新しい順）
                pth_files.sort(reverse=True)
                for i, f in enumerate(pth_files[:10], 1):
                    full_path = os.path.join(model_dir, f)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    print(f"  {i}. {f} ({size_mb:.1f} MB)")

                if len(pth_files) > 10:
                    print(f"  ... and {len(pth_files) - 10} more files")
            else:
                print(f"No .pth files found in {model_dir}")
        else:
            print(f"Directory not found: {model_dir}")

        print(f"\nUsage: python {sys.argv[0]} <model_path>")
        print(f"Example: python {sys.argv[0]} models/model_name.pth")
        sys.exit(1)

    # デフォルトパス設定
    base_path = os.path.splitext(args.model_path)[0]
    onnx_path = args.onnx or f"{base_path}.onnx"
    engine_path = args.engine or f"{base_path}.trt"

    # ONNX変換
    onnx_path = convert_pytorch_to_onnx(args.model_path, onnx_path, args.model_type)

    # TensorRTエンジンビルド
    engine_path = build_engine_from_onnx(onnx_path, engine_path, fp16_mode=not args.no_fp16)

    # ベンチマーク（オプション）
    if args.benchmark and engine_path:
        benchmark_model(args.model_path, engine_path)

    print(f"\nConversion completed!")
    print(f"ONNX model: {onnx_path}")
    print(f"TensorRT engine: {engine_path}")
