#!/usr/bin/env python3
"""
Comparación de Calidad FP32 vs INT8

Compara outputs de modelos FP32 e INT8 para validar que la cuantización
no degrada significativamente la calidad.

Métricas:
- Similitud coseno (objetivo: >0.98)
- MSE (Mean Squared Error, objetivo: <0.01)
- MAE (Mean Absolute Error, objetivo: <0.05)

Usage:
    python3 scripts/compare_fp32_int8_quality.py
"""

import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Tuple
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class QualityComparator:
    """Comparador de calidad FP32 vs INT8"""
    
    def __init__(self):
        self.fp32_path = project_root / "models/onnx/agi_audio_core.onnx"
        self.int8_path = project_root / "models/onnx/agi_audio_core_int8.onnx"
        
        # Verificar modelos existen
        if not self.fp32_path.exists():
            raise FileNotFoundError(f"Modelo FP32 no encontrado: {self.fp32_path}")
        if not self.int8_path.exists():
            raise FileNotFoundError(f"Modelo INT8 no encontrado: {self.int8_path}")
        
        print("=" * 60)
        print("SARAi v2.16.1 - Comparación Calidad FP32 vs INT8")
        print("=" * 60)
        print()
    
    def load_models(self) -> Tuple[ort.InferenceSession, ort.InferenceSession]:
        """Carga ambos modelos ONNX"""
        print("[1/5] Cargando modelos...")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        start = time.time()
        fp32_session = ort.InferenceSession(
            str(self.fp32_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        fp32_time = time.time() - start
        print(f"  ✅ FP32 cargado: {fp32_time:.2f}s")
        
        start = time.time()
        int8_session = ort.InferenceSession(
            str(self.int8_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        int8_time = time.time() - start
        print(f"  ✅ INT8 cargado: {int8_time:.2f}s")
        print(f"  📊 Speedup carga: {fp32_time/int8_time:.2f}x")
        print()
        
        return fp32_session, int8_session
    
    def generate_test_inputs(self, num_samples: int = 5) -> np.ndarray:
        """Genera inputs de test deterministas"""
        print(f"[2/5] Generando {num_samples} inputs de test...")
        
        np.random.seed(42)  # Reproducibilidad
        inputs = []
        
        for i in range(num_samples):
            # Audio codes aleatorios pero deterministas
            audio_codes = np.random.randint(
                low=0,
                high=1024,
                size=(1, 16, 128),
                dtype=np.int64
            )
            inputs.append(audio_codes)
        
        print(f"  ✅ {num_samples} inputs generados")
        print(f"  📐 Shape: (1, 16, 128)")
        print()
        
        return inputs
    
    def run_inference(
        self,
        fp32_session: ort.InferenceSession,
        int8_session: ort.InferenceSession,
        inputs: list
    ) -> Tuple[list, list, Dict]:
        """Ejecuta inferencia en ambos modelos"""
        print("[3/5] Ejecutando inferencias...")
        
        fp32_outputs = []
        int8_outputs = []
        fp32_times = []
        int8_times = []
        
        for i, audio_codes in enumerate(inputs):
            # FP32 inference
            start = time.time()
            fp32_result = fp32_session.run(
                None,
                {"audio_codes": audio_codes}
            )[0]
            fp32_time = time.time() - start
            fp32_outputs.append(fp32_result)
            fp32_times.append(fp32_time)
            
            # INT8 inference
            start = time.time()
            int8_result = int8_session.run(
                None,
                {"audio_codes": audio_codes}
            )[0]
            int8_time = time.time() - start
            int8_outputs.append(int8_result)
            int8_times.append(int8_time)
            
            print(f"  Sample {i+1}/{len(inputs)}: "
                  f"FP32={fp32_time:.2f}s, INT8={int8_time:.2f}s "
                  f"(Speedup: {fp32_time/int8_time:.2f}x)")
        
        # Stats de timing
        timing_stats = {
            "fp32_avg": np.mean(fp32_times),
            "int8_avg": np.mean(int8_times),
            "fp32_std": np.std(fp32_times),
            "int8_std": np.std(int8_times),
            "speedup_avg": np.mean(fp32_times) / np.mean(int8_times)
        }
        
        print()
        print(f"  📊 Latencia promedio:")
        print(f"     FP32: {timing_stats['fp32_avg']:.2f}s (±{timing_stats['fp32_std']:.2f}s)")
        print(f"     INT8: {timing_stats['int8_avg']:.2f}s (±{timing_stats['int8_std']:.2f}s)")
        print(f"     Speedup: {timing_stats['speedup_avg']:.2f}x")
        print()
        
        return fp32_outputs, int8_outputs, timing_stats
    
    def compute_quality_metrics(
        self,
        fp32_outputs: list,
        int8_outputs: list
    ) -> Dict[str, float]:
        """Calcula métricas de calidad"""
        print("[4/5] Calculando métricas de calidad...")
        
        cosine_sims = []
        mses = []
        maes = []
        
        for i, (fp32_out, int8_out) in enumerate(zip(fp32_outputs, int8_outputs)):
            # Aplanar arrays para comparación
            fp32_flat = fp32_out.flatten()
            int8_flat = int8_out.flatten()
            
            # Similitud coseno
            dot_product = np.dot(fp32_flat, int8_flat)
            norm_fp32 = np.linalg.norm(fp32_flat)
            norm_int8 = np.linalg.norm(int8_flat)
            cosine_sim = dot_product / (norm_fp32 * norm_int8)
            cosine_sims.append(cosine_sim)
            
            # MSE (Mean Squared Error)
            mse = np.mean((fp32_flat - int8_flat) ** 2)
            mses.append(mse)
            
            # MAE (Mean Absolute Error)
            mae = np.mean(np.abs(fp32_flat - int8_flat))
            maes.append(mae)
            
            print(f"  Sample {i+1}: Cosine={cosine_sim:.4f}, MSE={mse:.6f}, MAE={mae:.4f}")
        
        metrics = {
            "cosine_sim_avg": np.mean(cosine_sims),
            "cosine_sim_std": np.std(cosine_sims),
            "mse_avg": np.mean(mses),
            "mse_std": np.std(mses),
            "mae_avg": np.mean(maes),
            "mae_std": np.std(maes)
        }
        
        print()
        return metrics
    
    def evaluate_metrics(self, metrics: Dict, timing_stats: Dict) -> bool:
        """Evalúa si las métricas cumplen objetivos"""
        print("[5/5] Evaluación de métricas...")
        print()
        
        passed = True
        
        # Objetivos de calidad
        objectives = {
            "cosine_sim": {"value": metrics["cosine_sim_avg"], "target": 0.98, "operator": ">="},
            "mse": {"value": metrics["mse_avg"], "target": 0.01, "operator": "<="},
            "mae": {"value": metrics["mae_avg"], "target": 0.05, "operator": "<="},
            "speedup": {"value": timing_stats["speedup_avg"], "target": 2.0, "operator": ">="}
        }
        
        print("━" * 60)
        print("RESULTADOS DE CALIDAD")
        print("━" * 60)
        
        for metric_name, obj in objectives.items():
            value = obj["value"]
            target = obj["target"]
            operator = obj["operator"]
            
            if operator == ">=":
                success = value >= target
                status = "✅" if success else "❌"
                comparison = f"{value:.4f} >= {target:.4f}"
            else:  # "<="
                success = value <= target
                status = "✅" if success else "❌"
                comparison = f"{value:.4f} <= {target:.4f}"
            
            print(f"{status} {metric_name.upper():12s}: {comparison}")
            
            if not success:
                passed = False
        
        print("━" * 60)
        
        if passed:
            print("✅ CALIDAD VALIDADA: INT8 cumple todos los objetivos")
        else:
            print("⚠️  ADVERTENCIA: INT8 no cumple algunos objetivos")
            print("   Considerar:")
            print("   - Probar FP16 (mejor precisión)")
            print("   - Usar Static Quantization con calibración")
            print("   - Verificar dataset de calibración")
        
        print()
        
        # Resumen de beneficios
        print("━" * 60)
        print("RESUMEN DE BENEFICIOS")
        print("━" * 60)
        
        fp32_size = 4.3  # GB
        int8_size = 1.1  # GB
        size_reduction = (fp32_size - int8_size) / fp32_size * 100
        
        print(f"📦 Tamaño:      {fp32_size:.1f}GB → {int8_size:.1f}GB (-{size_reduction:.0f}%)")
        print(f"⚡ Speedup:     {timing_stats['speedup_avg']:.2f}x")
        print(f"🎯 Precisión:   {metrics['cosine_sim_avg']*100:.1f}%")
        print(f"💾 RAM ahorrada: ~{fp32_size - int8_size:.1f}GB")
        print("━" * 60)
        print()
        
        return passed
    
    def run(self) -> bool:
        """Ejecuta comparación completa"""
        try:
            # 1. Cargar modelos
            fp32_session, int8_session = self.load_models()
            
            # 2. Generar inputs
            inputs = self.generate_test_inputs(num_samples=5)
            
            # 3. Inferencia
            fp32_outputs, int8_outputs, timing_stats = self.run_inference(
                fp32_session,
                int8_session,
                inputs
            )
            
            # 4. Métricas de calidad
            quality_metrics = self.compute_quality_metrics(
                fp32_outputs,
                int8_outputs
            )
            
            # 5. Evaluación
            passed = self.evaluate_metrics(quality_metrics, timing_stats)
            
            return passed
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    comparator = QualityComparator()
    success = comparator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
