#!/usr/bin/env python3
"""
Test del modelo qwen25_7b_audio.onnx (42MB)
Comparar con qwen25_audio.onnx (385MB)
"""

import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Colores
class C:
    G = '\033[92m'
    Y = '\033[93m'
    R = '\033[91m'
    B = '\033[94m'
    M = '\033[95m'
    E = '\033[0m'


def test_model(model_path: Path, model_name: str):
    """Test un modelo ONNX"""
    
    print(f"\n{C.B}━━━ Testing: {model_name} ━━━{C.E}")
    print(f"Archivo: {model_path}")
    
    if not model_path.exists():
        print(f"{C.R}✗{C.E} Archivo no encontrado")
        return None
    
    # Tamaño
    size_mb = model_path.stat().st_size / (1024**2)
    data_file = model_path.with_suffix('.onnx.data')
    if data_file.exists():
        size_mb += data_file.stat().st_size / (1024**2)
    
    print(f"Tamaño total: {size_mb:.1f} MB")
    
    # Cargar
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    start = time.perf_counter()
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        load_time = (time.perf_counter() - start) * 1000
        print(f"{C.G}✓{C.E} Cargado en {load_time:.1f}ms")
    except Exception as e:
        print(f"{C.R}✗{C.E} Error al cargar: {e}")
        return None
    
    # Metadata
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    print(f"\n{C.M}Metadata:{C.E}")
    print(f"  Input:  {input_info.name}")
    print(f"          Shape: {input_info.shape}")
    print(f"          Type:  {input_info.type}")
    print(f"  Output: {output_info.name}")
    print(f"          Shape: {output_info.shape}")
    print(f"          Type:  {output_info.type}")
    
    # Determinar shape real
    input_shape = input_info.shape
    if isinstance(input_shape[1], str) or isinstance(input_shape[2], str):
        # Shape dinámica, usar defaults
        batch_size = 1
        seq_len = 128
        if len(input_shape) == 3:
            hidden_dim = int(input_shape[2]) if not isinstance(input_shape[2], str) else 3584
        else:
            hidden_dim = 3584
    else:
        batch_size = 1
        seq_len = input_shape[1]
        hidden_dim = input_shape[2]
    
    print(f"\n{C.M}Shape para test:{C.E}")
    print(f"  Usando: [1, {seq_len}, {hidden_dim}]")
    
    # Generar datos sintéticos
    if 'float16' in input_info.type or 'Float16' in input_info.type:
        dtype = np.float16
        print(f"  Dtype: float16")
    else:
        dtype = np.float32
        print(f"  Dtype: float32")
    
    hidden_states = np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
    hidden_states = hidden_states * 0.1
    hidden_states = np.clip(hidden_states, -0.5, 0.5)
    hidden_states = hidden_states.astype(dtype)
    
    # Inference (10 runs)
    latencies = []
    output = None
    
    print(f"\n{C.M}Inference (10 runs):{C.E}")
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    for i in range(10):
        start = time.perf_counter()
        try:
            output = session.run(
                [output_name],
                {input_name: hidden_states}
            )[0]
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
            print(f"  Run {i+1}/10: {latency:6.1f}ms", end='\r')
        except Exception as e:
            print(f"\n{C.R}✗{C.E} Error en run {i+1}: {e}")
            latencies.append(0)
    
    print()
    
    # Resultados
    valid_latencies = [lat for lat in latencies if lat > 0]
    
    if valid_latencies:
        print(f"\n{C.M}Resultados:{C.E}")
        print(f"  Min:    {min(valid_latencies):6.1f}ms")
        print(f"  P50:    {np.percentile(valid_latencies, 50):6.1f}ms")
        print(f"  P95:    {np.percentile(valid_latencies, 95):6.1f}ms")
        print(f"  Max:    {max(valid_latencies):6.1f}ms")
        print(f"  Avg:    {np.mean(valid_latencies):6.1f}ms")
        
        if output is not None:
            print(f"\n  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
            print(f"  Output min:   {output.min():.4f}")
            print(f"  Output max:   {output.max():.4f}")
    
    return {
        'load_time': load_time,
        'latencies': valid_latencies,
        'size_mb': size_mb,
        'output_shape': output.shape if output is not None else None
    }


def main():
    print(f"\n{C.M}{'='*70}{C.E}")
    print(f"{C.M}  Comparación de Modelos Talker ONNX  {C.E}")
    print(f"{C.M}{'='*70}{C.E}")
    
    base_path = Path(__file__).parent.parent
    
    models = {
        "qwen25_7b (42MB)": base_path / "models/onnx/qwen25_7b_audio.onnx",
        "qwen25 (385MB)": base_path / "models/onnx/old/qwen25_audio.onnx",
    }
    
    results = {}
    
    for name, path in models.items():
        result = test_model(path, name)
        if result:
            results[name] = result
    
    # Comparación
    if len(results) >= 2:
        print(f"\n{C.B}{'='*70}{C.E}")
        print(f"{C.B}  COMPARACIÓN FINAL  {C.E}")
        print(f"{C.B}{'='*70}{C.E}")
        
        print(f"\n{'Métrica':<20} {'qwen25_7b (42MB)':<20} {'qwen25 (385MB)':<20}")
        print(f"{'-'*60}")
        
        for name in results.keys():
            r = results[name]
            model_short = name.split()[0]
            
            print(f"Tamaño:              {r['size_mb']:.1f} MB")
            print(f"Carga:               {r['load_time']:.1f}ms")
            print(f"Latencia P50:        {np.percentile(r['latencies'], 50):.1f}ms")
            print(f"Latencia P95:        {np.percentile(r['latencies'], 95):.1f}ms")
            print(f"Output shape:        {r['output_shape']}")
            print()
    
    print(f"{C.G}{'='*70}{C.E}")
    print(f"{C.G}✓ Comparación completa{C.E}")
    print(f"{C.G}{'='*70}{C.E}")


if __name__ == "__main__":
    main()
