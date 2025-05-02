import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import create_model
from optimizer import create_onnx_session
import os

def benchmark_pytorch(model, input_size, num_iterations=100):
    model.eval()
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            input_data = torch.randn(input_size)
            start_time = time.time()
            _ = model(input_data)
            end_time = time.time()
            times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def benchmark_onnx(session, input_size, num_iterations=100):
    times = []
    
    for _ in range(num_iterations):
        input_data = np.random.randn(*input_size).astype(np.float32)
        start_time = time.time()
        _ = session.run(None, {'input': input_data})
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def plot_results(results, title='Inference Time Comparison'):
    plt.figure(figsize=(10, 6))
    frameworks = list(results.keys())
    means = [results[f]['mean'] * 1000 for f in frameworks]  # Convert to ms
    stds = [results[f]['std'] * 1000 for f in frameworks]
    
    plt.bar(frameworks, means, yerr=stds, capsize=5)
    plt.title(title)
    plt.ylabel('Inference Time (ms)')
    plt.grid(True, alpha=0.3)
    plt.savefig('benchmark_results.png')
    plt.close()

def main():
    # Set number of threads for fair comparison
    num_threads = os.cpu_count()
    torch.set_num_threads(num_threads)
    print(f"Benchmarking with {num_threads} threads")

    # Load models
    input_size = (1, 3, 224, 224)
    pytorch_model = create_model(pretrained=False)
    pytorch_model.eval()
    
    onnx_session = create_onnx_session('vit_model.onnx', num_threads=num_threads)

    # Run benchmarks
    print("Running PyTorch benchmark...")
    pytorch_mean, pytorch_std = benchmark_pytorch(pytorch_model, input_size)
    
    print("Running ONNX Runtime benchmark...")
    onnx_mean, onnx_std = benchmark_onnx(onnx_session, input_size)

    # Collect and display results
    results = {
        'PyTorch': {'mean': pytorch_mean, 'std': pytorch_std},
        'ONNX Runtime': {'mean': onnx_mean, 'std': onnx_std}
    }

    print("\nBenchmark Results:")
    print("-" * 50)
    for framework, metrics in results.items():
        print(f"{framework}:")
        print(f"  Mean inference time: {metrics['mean']*1000:.2f} ms")
        print(f"  Standard deviation: {metrics['std']*1000:.2f} ms")

    # Plot results
    plot_results(results)
    print("\nResults have been plotted to 'benchmark_results.png'")

if __name__ == '__main__':
    main()
