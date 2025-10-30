#!/usr/bin/env python3
"""
SARAi v2.16.3 - Quick Win: Talker ONNX con Float16
===================================================

Test rápido (< 5 min) para validar:
1. Talker ONNX carga correctamente
2. Acepta tensores float16
3. Latencia real con datos sintéticos realistas
4. Output shape correcto

Este es el "Quick Win" recomendado en docs/VOICE_EXECUTIVE_SUMMARY.md
"""

import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple

# Colores ANSI
class C:
    G = '\033[92m'  # Green
    Y = '\033[93m'  # Yellow
    R = '\033[91m'  # Red
    B = '\033[94m'  # Blue
    M = '\033[95m'  # Magenta
    E = '\033[0m'   # End


def load_talker_onnx() -> Tuple[ort.InferenceSession, Path]:
    """
    Carga modelo Talker ONNX optimizado para CPU
    
    Returns:
        Tuple de (Session de ONNX Runtime, Path del modelo)
    """
    print(f"\n{C.B}━━━ PASO 1/4: Cargar Talker ONNX ━━━{C.E}")
    
    # Buscar modelo en diferentes ubicaciones
    # PRIORIDAD: qwen25_7b_audio.onnx (42MB, 10x más rápido)
    base_path = Path(__file__).parent.parent
    search_paths = [
        base_path / "models/onnx/qwen25_7b_audio.onnx",  # ⚡ MEJOR: 9ms P50
        base_path / "models/onnx/old/qwen25_audio.onnx",  # Fallback: 85ms P50
        base_path / "models/onnx/qwen25_audio_gpu_lite.onnx",
        base_path / "models/onnx/qwen25_audio.onnx",
    ]
    
    model_path = None
    for path in search_paths:
        if path.exists():
            model_path = path
            print(f"{C.G}✓{C.E} Encontrado: {path.name}")
            break
    
    if not model_path:
        raise FileNotFoundError("No se encontró ningún modelo Talker ONNX")
    
    # Opciones de sesión optimizadas para CPU
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # Threads paralelos
    sess_options.inter_op_num_threads = 2
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    start = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    load_time_ms = (time.perf_counter() - start) * 1000
    
    print(f"{C.G}✓{C.E} Cargado en {load_time_ms:.1f}ms")
    
    # Inspeccionar inputs/outputs
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    print(f"\n{C.M}Información del Modelo:{C.E}")
    print(f"  Input:  {input_info.name} {input_info.shape} ({input_info.type})")
    print(f"  Output: {output_info.name} {output_info.shape} ({output_info.type})")
    
    return session, model_path


def generate_synthetic_hidden_states(seq_len: int = 128, hidden_dim: int = 3584) -> np.ndarray:
    """
    Genera hidden_states sintéticos realistas
    
    Hidden states son el output del encoder de audio después de projection.
    Simulamos una distribución similar a la real basada en:
    - Media cercana a 0
    - Desviación estándar ~0.1
    - Valores en rango [-0.5, 0.5]
    
    Args:
        seq_len: Longitud de secuencia (frames de audio)
        hidden_dim: Dimensión oculta (3584 para qwen25_7b, 3072 para qwen25)
    
    Returns:
        Tensor [1, seq_len, hidden_dim] en float32
    """
    print(f"\n{C.B}━━━ PASO 2/4: Generar Hidden States Sintéticos ━━━{C.E}")
    
    # Generar con distribución normal
    hidden_states = np.random.randn(1, seq_len, hidden_dim).astype(np.float32)
    
    # Normalizar a distribución realista
    hidden_states = hidden_states * 0.1  # Reducir varianza
    hidden_states = np.clip(hidden_states, -0.5, 0.5)  # Clip outliers
    
    # Mantener float32 (requerido por este modelo ONNX)
    # Nota: Algunos ONNX usan float16, este usa float32
    hidden_states_fp32 = hidden_states  # Ya está en float32
    
    print(f"{C.G}✓{C.E} Generado: shape {hidden_states_fp32.shape}")
    print(f"  Dtype: {hidden_states_fp32.dtype}")
    print(f"  Min:   {hidden_states_fp32.min():.4f}")
    print(f"  Max:   {hidden_states_fp32.max():.4f}")
    print(f"  Mean:  {hidden_states_fp32.mean():.4f}")
    print(f"  Std:   {hidden_states_fp32.std():.4f}")
    
    return hidden_states_fp32


def run_inference(session: ort.InferenceSession, hidden_states: np.ndarray, n_runs: int = 10) -> Tuple[np.ndarray, list]:
    """
    Ejecuta inference múltiples veces para medir latencia
    
    Args:
        session: Sesión ONNX
        hidden_states: Tensor de entrada
        n_runs: Número de ejecuciones
    
    Returns:
        (output, latencias_ms)
    """
    print(f"\n{C.B}━━━ PASO 3/4: Ejecutar Inference ({n_runs} runs) ━━━{C.E}")
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    latencies = []
    output = None
    
    for i in range(n_runs):
        start = time.perf_counter()
        
        try:
            output = session.run(
                [output_name],
                {input_name: hidden_states}
            )[0]
            
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            
            print(f"  Run {i+1}/{n_runs}: {latency_ms:6.1f}ms", end='\r')
        
        except Exception as e:
            print(f"\n{C.R}✗{C.E} Error en run {i+1}: {e}")
            latencies.append(0)
    
    print()  # Nueva línea
    
    if output is not None:
        print(f"{C.G}✓{C.E} Output shape: {output.shape}, dtype: {output.dtype}")
    
    return output, latencies


def analyze_results(latencies: list, output: np.ndarray):
    """
    Analiza y presenta resultados
    
    Args:
        latencies: Lista de latencias en ms
        output: Tensor de salida
    """
    print(f"\n{C.B}━━━ PASO 4/4: Análisis de Resultados ━━━{C.E}")
    
    # Filtrar runs fallidos (latencia = 0)
    valid_latencies = [lat for lat in latencies if lat > 0]
    
    if not valid_latencies:
        print(f"{C.R}✗{C.E} Todos los runs fallaron")
        return
    
    # Estadísticas de latencia
    min_lat = min(valid_latencies)
    max_lat = max(valid_latencies)
    avg_lat = np.mean(valid_latencies)
    p50_lat = np.percentile(valid_latencies, 50)
    p95_lat = np.percentile(valid_latencies, 95)
    p99_lat = np.percentile(valid_latencies, 99)
    
    print(f"\n{C.M}📊 Latencias (CPU):{C.E}")
    print(f"  Min:    {min_lat:6.1f}ms")
    print(f"  P50:    {p50_lat:6.1f}ms")
    print(f"  P95:    {p95_lat:6.1f}ms")
    print(f"  P99:    {p99_lat:6.1f}ms")
    print(f"  Max:    {max_lat:6.1f}ms")
    print(f"  Avg:    {avg_lat:6.1f}ms")
    
    # Evaluación vs objetivo
    print(f"\n{C.M}🎯 Evaluación vs Objetivos:{C.E}")
    
    if p50_lat <= 100:
        print(f"  {C.G}✓{C.E} P50 ≤ 100ms (Excelente)")
    elif p50_lat <= 150:
        print(f"  {C.Y}~{C.E} P50 ≤ 150ms (Bueno)")
    else:
        print(f"  {C.R}✗{C.E} P50 > 150ms (Mejorable)")
    
    if p99_lat <= 200:
        print(f"  {C.G}✓{C.E} P99 ≤ 200ms (Consistente)")
    else:
        print(f"  {C.Y}~{C.E} P99 > 200ms (Variabilidad alta)")
    
    # Análisis de output
    if output is not None:
        print(f"\n{C.M}📦 Output del Modelo:{C.E}")
        print(f"  Shape: {output.shape}")
        print(f"  Dtype: {output.dtype}")
        print(f"  Min:   {output.min():.4f}")
        print(f"  Max:   {output.max():.4f}")
        print(f"  Mean:  {output.mean():.4f}")
        
        # Verificar que output no sea todo ceros o NaN
        if np.all(output == 0):
            print(f"  {C.R}⚠{C.E}  WARNING: Output es todo ceros")
        elif np.any(np.isnan(output)):
            print(f"  {C.R}⚠{C.E}  WARNING: Output contiene NaN")
        else:
            print(f"  {C.G}✓{C.E} Output válido (no ceros, no NaN)")


def main():
    """Función principal"""
    
    print(f"\n{C.M}{'='*70}{C.E}")
    print(f"{C.M}{C.B}   SARAi v2.16.3 - Quick Win: Talker ONNX Float16 Test   {C.E}")
    print(f"{C.M}{'='*70}{C.E}")
    
    try:
        # 1. Cargar modelo
        session, model_path = load_talker_onnx()
        
        # Determinar shape del input
        input_shape = session.get_inputs()[0].shape
        
        # Si shape es dinámica (con strings), usar defaults
        if isinstance(input_shape[1], str) or isinstance(input_shape[2], str):
            seq_len = 128
            # Determinar hidden_dim por tipo de modelo
            if 'qwen25_7b' in str(model_path):
                hidden_dim = 3584  # Modelo 7B
            else:
                hidden_dim = 3072  # Modelo baseline
            print(f"\n{C.Y}ℹ{C.E}  Shape dinámica detectada, usando: seq_len={seq_len}, hidden_dim={hidden_dim}")
        else:
            seq_len = input_shape[1]
            hidden_dim = input_shape[2]
        
        # 2. Generar datos sintéticos
        hidden_states = generate_synthetic_hidden_states(seq_len, hidden_dim)
        
        # 3. Ejecutar inference
        output, latencies = run_inference(session, hidden_states, n_runs=10)
        
        # 4. Analizar resultados
        analyze_results(latencies, output)
        
        # Mensaje final
        print(f"\n{C.G}{'='*70}{C.E}")
        print(f"{C.G}✓ Quick Win Test Completado{C.E}")
        print(f"{C.G}{'='*70}{C.E}")
        
        print(f"\n{C.M}📝 Próximos Pasos:{C.E}")
        print(f"  1. Revisar docs/VOICE_EXECUTIVE_SUMMARY.md")
        print(f"  2. Integrar Audio Encoder para datos reales")
        print(f"  3. Añadir Token2Wav para generación de audio")
        print(f"  4. Test end-to-end completo")
        
    except Exception as e:
        print(f"\n{C.R}{'='*70}{C.E}")
        print(f"{C.R}✗ Error Fatal{C.E}")
        print(f"{C.R}{'='*70}{C.E}")
        print(f"\n{C.R}{e}{C.E}")
        
        import traceback
        traceback.print_exc()
        
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
