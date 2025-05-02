import torch
import onnx
import onnxruntime as ort
import numpy as np

def convert_to_onnx(model, sample_input, save_path='model.onnx'):
    """Convert PyTorch model to ONNX format"""
    torch.onnx.export(model,
                     sample_input,
                     save_path,
                     opset_version=12,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})
    
    # Optimize the ONNX model
    model = onnx.load(save_path)
    optimized_model = onnx.optimizer.optimize(model)
    onnx.save(optimized_model, save_path)

def create_onnx_session(onnx_path, num_threads=None):
    """Create an optimized ONNX Runtime session"""
    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    
    if num_threads is not None:
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
    
    # Enable OpenMP optimizations
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    return ort.InferenceSession(onnx_path, 
                              providers=providers,
                              sess_options=sess_options)

def inference_onnx(session, input_data):
    """Run inference using ONNX Runtime"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: input_data})[0]
