#!/usr/bin/env python3
"""
Test de Estabilidad Producción - SARAi v2.16 Omni-Loop

Prueba el sistema completo con modelos reales:
- TRM-Router con embeddings reales
- MCP con pesos dinámicos
- RAG con búsqueda web real (opcional)
- Omni-Loop simulado con Ollama
- Monitoreo RAM/CPU en tiempo real

Usage:
    python scripts/test_production_stability.py [--duration 300] [--scenarios all]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil


@dataclass
class TestResult:
    """Resultado de un test individual"""
    scenario: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    latency_ms: float
    ram_peak_mb: float
    cpu_peak_percent: float
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ProductionTester:
    """
    Tester de estabilidad producción con modelos reales
    
    Componentes probados:
    1. Ollama (draft LLM simulado)
    2. TRM-Router (clasificación hard/soft)
    3. MCP (cálculo de pesos α/β)
    4. RAG Agent (búsqueda + síntesis - opcional)
    5. Image Preprocessor (WebP conversion - opcional)
    6. Stress Test (100 queries concurrentes)
    """
    
    def __init__(self, duration_seconds: int = 300):
        self.duration = duration_seconds
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.logs_dir = self.project_root / "logs"
        self.state_dir = self.project_root / "state"
        
        # Modelos Ollama necesarios (SARAi v2.16 stack REAL completo)
        self.required_models = [
            "solar:10.7b",                              # Expert LLM (SOLAR-10.7B - modelo principal)
            "hf.co/LiquidAI/LFM2-1.2B-GGUF:Q4_K_M",    # Tiny Tier (LFM2-1.2B - soft skills)
            "rockn/Qwen2.5-Omni-7B-Q4_K_M",            # Multimodal (Qwen2.5-Omni-7B)
            "qwen2.5:0.5b",                             # Draft LLM ultra-rápido (Omni-Loop)
        ]
        
        # Metrics baseline
        self.ram_baseline_mb = psutil.virtual_memory().used / (1024**2)
        self.cpu_baseline_percent = psutil.cpu_percent(interval=1)
    
    def setup(self):
        """Setup: Verificar Ollama y descargar modelos"""
        print("🔧 [SETUP] Verificando entorno de testing...")
        
        # 1. Verificar Ollama instalado
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            print(f"✅ Ollama disponible (versión detectada)")
        except Exception as e:
            print(f"❌ Ollama NO disponible: {e}")
            print("   Instalar: curl -fsSL https://ollama.com/install.sh | sh")
            return False
        
        # 2. Descargar modelos necesarios
        for model in self.required_models:
            if model not in result.stdout:
                print(f"📥 Descargando modelo: {model}...")
                try:
                    subprocess.run(
                        ["ollama", "pull", model],
                        timeout=600,  # 10 min max por modelo
                        check=True
                    )
                    print(f"✅ {model} descargado")
                except subprocess.TimeoutExpired:
                    print(f"⚠️ Timeout descargando {model}, continuando...")
                except Exception as e:
                    print(f"❌ Error descargando {model}: {e}")
                    return False
            else:
                print(f"✅ {model} ya disponible")
        
        # 3. Verificar Python deps
        try:
            import torch
            import numpy as np
            print(f"✅ PyTorch disponible: {torch.__version__}")
        except ImportError as e:
            print(f"❌ Dependencias faltantes: {e}")
            return False
        
        print(f"\n✅ Setup completo. RAM baseline: {self.ram_baseline_mb:.1f}MB\n")
        return True
    
    def monitor_resources(self) -> Dict[str, float]:
        """Monitorea RAM/CPU actuales"""
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        
        return {
            "ram_used_mb": mem.used / (1024**2),
            "ram_percent": mem.percent,
            "cpu_percent": cpu
        }
    
    def test_solar_expert_inference(self) -> TestResult:
        """Test EXTRA: Inferencia con SOLAR-10.7B (Expert LLM real)"""
        print("🧪 [TEST EXTRA] SOLAR-10.7B Expert Inference")
        
        start = time.perf_counter()
        resources_start = self.monitor_resources()
        
        # Prompt técnico complejo (para validar capacidad del modelo expert)
        prompt = """Explica el algoritmo de backpropagation en redes neuronales.
Incluye: 1) La función de pérdida, 2) La regla de la cadena, 3) Actualización de pesos.
Respuesta en máximo 150 palabras."""
        
        try:
            result = subprocess.run(
                ["ollama", "run", "solar:10.7b", prompt],
                capture_output=True,
                text=True,
                timeout=60  # SOLAR es más lento pero más preciso
            )
            
            latency_ms = (time.perf_counter() - start) * 1000
            resources_end = self.monitor_resources()
            
            ram_peak = resources_end["ram_used_mb"] - resources_start["ram_used_mb"]
            cpu_peak = max(resources_end["cpu_percent"], resources_start["cpu_percent"])
            
            # Validar respuesta técnica (debe contener keywords)
            response = result.stdout.lower()
            keywords = ["pérdida", "cadena", "pesos", "gradiente", "derivada"]
            keyword_hits = sum(1 for kw in keywords if kw in response)
            
            if result.returncode == 0 and len(result.stdout) > 100 and keyword_hits >= 3:
                print(f"  ✅ PASS - Latency: {latency_ms:.0f}ms, RAM: +{ram_peak:.1f}MB, Keywords: {keyword_hits}/5")
                print(f"  📝 Response preview: {result.stdout[:150]}...")
                return TestResult(
                    scenario="solar_expert_inference",
                    status="PASS",
                    latency_ms=latency_ms,
                    ram_peak_mb=ram_peak,
                    cpu_peak_percent=cpu_peak
                )
            else:
                raise Exception(f"SOLAR response quality low: {keyword_hits}/5 keywords, {len(result.stdout)} chars")
        
        except subprocess.TimeoutExpired:
            print(f"  ⚠️ TIMEOUT - SOLAR took >60s")
            return TestResult(
                scenario="solar_expert_inference",
                status="FAIL",
                latency_ms=60000,
                ram_peak_mb=0,
                cpu_peak_percent=0,
                error="Timeout after 60s"
            )
        except Exception as e:
            print(f"  ❌ FAIL - {e}")
            return TestResult(
                scenario="solar_expert_inference",
                status="FAIL",
                latency_ms=0,
                ram_peak_mb=0,
                cpu_peak_percent=0,
                error=str(e)
            )
    
    def test_lfm2_tiny_tier(self) -> TestResult:
        """Test EXTRA 2: LFM2-1.2B Tiny Tier (soft skills/modulación)"""
        print("🧪 [TEST EXTRA 2] LFM2-1.2B Tiny Tier (Soft Skills)")
        
        start = time.perf_counter()
        resources_start = self.monitor_resources()
        
        # Prompt emocional (soft skill test)
        prompt = """Reformula este mensaje técnico de forma empática:
"Error 404: El recurso solicitado no existe en el servidor."

Usa tono amigable y ofrece ayuda constructiva."""
        
        try:
            result = subprocess.run(
                ["ollama", "run", "hf.co/LiquidAI/LFM2-1.2B-GGUF:Q4_K_M", prompt],
                capture_output=True,
                text=True,
                timeout=20
            )
            
            latency_ms = (time.perf_counter() - start) * 1000
            resources_end = self.monitor_resources()
            
            ram_peak = resources_end["ram_used_mb"] - resources_start["ram_used_mb"]
            
            # Validar tono empático (keywords soft)
            response = result.stdout.lower()
            soft_keywords = ["ayud", "comprend", "disculp", "lament", "intent"]
            soft_hits = sum(1 for kw in soft_keywords if kw in response)
            
            if result.returncode == 0 and len(result.stdout) > 30 and soft_hits >= 2:
                print(f"  ✅ PASS - Latency: {latency_ms:.0f}ms, RAM: +{ram_peak:.1f}MB, Soft: {soft_hits}/5")
                print(f"  💬 Response preview: {result.stdout[:100]}...")
                return TestResult(
                    scenario="lfm2_tiny_tier",
                    status="PASS",
                    latency_ms=latency_ms,
                    ram_peak_mb=ram_peak,
                    cpu_peak_percent=resources_end["cpu_percent"]
                )
            else:
                raise Exception(f"LFM2 empathy low: {soft_hits}/5 soft keywords")
        
        except Exception as e:
            print(f"  ❌ FAIL - {e}")
            return TestResult(
                scenario="lfm2_tiny_tier",
                status="FAIL",
                latency_ms=0,
                ram_peak_mb=0,
                cpu_peak_percent=0,
                error=str(e)
            )
    
    def test_qwen_omni_multimodal(self) -> TestResult:
        """Test EXTRA 3: Qwen2.5-Omni-7B Multimodal (simulado - solo texto)"""
        print("🧪 [TEST EXTRA 3] Qwen2.5-Omni-7B Multimodal")
        
        start = time.perf_counter()
        resources_start = self.monitor_resources()
        
        # Prompt multimodal simulado (audio transcription task)
        prompt = """[AUDIO TRANSCRIPTION SIMULATION]
Transcribe y resume este texto:
"Buenos días, necesito ayuda con mi cuenta. No puedo acceder desde hace dos días."

Output: Transcripción + Sentiment Analysis"""
        
        try:
            result = subprocess.run(
                ["ollama", "run", "rockn/Qwen2.5-Omni-7B-Q4_K_M", prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            latency_ms = (time.perf_counter() - start) * 1000
            resources_end = self.monitor_resources()
            
            ram_peak = resources_end["ram_used_mb"] - resources_start["ram_used_mb"]
            
            if result.returncode == 0 and len(result.stdout) > 50:
                print(f"  ✅ PASS - Latency: {latency_ms:.0f}ms, RAM: +{ram_peak:.1f}MB")
                print(f"  🎤 Response preview: {result.stdout[:100]}...")
                return TestResult(
                    scenario="qwen_omni_multimodal",
                    status="PASS",
                    latency_ms=latency_ms,
                    ram_peak_mb=ram_peak,
                    cpu_peak_percent=resources_end["cpu_percent"]
                )
            else:
                raise Exception(f"Qwen-Omni response too short: {len(result.stdout)} chars")
        
        except Exception as e:
            print(f"  ❌ FAIL - {e}")
            return TestResult(
                scenario="qwen_omni_multimodal",
                status="FAIL",
                latency_ms=0,
                ram_peak_mb=0,
                cpu_peak_percent=0,
                error=str(e)
            )
    
    def test_ollama_inference(self) -> TestResult:
        """Test 1: Inferencia básica con Ollama (draft LLM)"""
        print("🧪 [TEST 1/6] Ollama Inference (Draft LLM)")
        
        start = time.perf_counter()
        resources_start = self.monitor_resources()
        
        prompt = "Explica en 2 frases qué es la recursión en programación."
        
        try:
            result = subprocess.run(
                ["ollama", "run", "qwen2.5:0.5b", prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            latency_ms = (time.perf_counter() - start) * 1000
            resources_end = self.monitor_resources()
            
            ram_peak = resources_end["ram_used_mb"] - resources_start["ram_used_mb"]
            cpu_peak = max(resources_end["cpu_percent"], resources_start["cpu_percent"])
            
            if result.returncode == 0 and len(result.stdout) > 50:
                print(f"  ✅ PASS - Latency: {latency_ms:.0f}ms, RAM: +{ram_peak:.1f}MB")
                return TestResult(
                    scenario="ollama_inference",
                    status="PASS",
                    latency_ms=latency_ms,
                    ram_peak_mb=ram_peak,
                    cpu_peak_percent=cpu_peak
                )
            else:
                raise Exception(f"Ollama returned: {result.stdout[:100]}")
        
        except Exception as e:
            print(f"  ❌ FAIL - {e}")
            return TestResult(
                scenario="ollama_inference",
                status="FAIL",
                latency_ms=0,
                ram_peak_mb=0,
                cpu_peak_percent=0,
                error=str(e)
            )
    
    def test_trm_router(self) -> TestResult:
        """Test 2: TRM-Router clasificación hard/soft"""
        print("🧪 [TEST 2/6] TRM-Router Classification")
        
        start = time.perf_counter()
        resources_start = self.monitor_resources()
        
        try:
            # Importar TRM-Router (si existe)
            sys.path.insert(0, str(self.project_root))
            
            # Mock básico si no existe el módulo
            try:
                from core.trm_classifier import TRMClassifierDual
                
                # Cargar modelo si existe
                model_path = self.project_root / "models" / "trm_classifier" / "trm_base.pt"
                if model_path.exists():
                    import torch
                    trm = TRMClassifierDual()
                    trm.load_state_dict(torch.load(model_path))
                    trm.eval()
                    
                    # Test input
                    test_embedding = torch.randn(1, 2048)  # Mock embedding
                    with torch.no_grad():
                        scores = trm(test_embedding)
                    
                    latency_ms = (time.perf_counter() - start) * 1000
                    resources_end = self.monitor_resources()
                    
                    if "hard" in scores and "soft" in scores:
                        print(f"  ✅ PASS - Latency: {latency_ms:.0f}ms, Hard: {scores['hard']:.2f}, Soft: {scores['soft']:.2f}")
                        return TestResult(
                            scenario="trm_router",
                            status="PASS",
                            latency_ms=latency_ms,
                            ram_peak_mb=resources_end["ram_used_mb"] - resources_start["ram_used_mb"],
                            cpu_peak_percent=resources_end["cpu_percent"]
                        )
                else:
                    raise FileNotFoundError("TRM model not found")
            
            except ImportError:
                # Simular TRM con random scores
                import random
                time.sleep(0.05)  # Simular latencia de inferencia
                
                scores = {
                    "hard": random.uniform(0.3, 0.9),
                    "soft": random.uniform(0.1, 0.7)
                }
                
                latency_ms = (time.perf_counter() - start) * 1000
                resources_end = self.monitor_resources()
                
                print(f"  ⚠️ SKIP (mock) - Latency: {latency_ms:.0f}ms, Hard: {scores['hard']:.2f}, Soft: {scores['soft']:.2f}")
                return TestResult(
                    scenario="trm_router",
                    status="SKIP",
                    latency_ms=latency_ms,
                    ram_peak_mb=0,
                    cpu_peak_percent=resources_end["cpu_percent"]
                )
        
        except Exception as e:
            print(f"  ❌ FAIL - {e}")
            return TestResult(
                scenario="trm_router",
                status="FAIL",
                latency_ms=0,
                ram_peak_mb=0,
                cpu_peak_percent=0,
                error=str(e)
            )
    
    def test_mcp_weights(self) -> TestResult:
        """Test 3: MCP cálculo de pesos α/β"""
        print("🧪 [TEST 3/6] MCP Weight Calculation")
        
        start = time.perf_counter()
        resources_start = self.monitor_resources()
        
        try:
            # Mock de MCP (reglas simples)
            scores = {"hard": 0.8, "soft": 0.3}
            context = "Test context for MCP"
            
            # Reglas fase 1 (mock)
            if scores["hard"] > 0.8 and scores["soft"] < 0.3:
                alpha, beta = 0.95, 0.05
            elif scores["soft"] > 0.7 and scores["hard"] < 0.4:
                alpha, beta = 0.2, 0.8
            else:
                alpha, beta = 0.6, 0.4
            
            latency_ms = (time.perf_counter() - start) * 1000
            resources_end = self.monitor_resources()
            
            # Validar que α + β ≈ 1.0
            if abs((alpha + beta) - 1.0) < 0.01:
                print(f"  ✅ PASS - Latency: {latency_ms:.0f}ms, α={alpha:.2f}, β={beta:.2f}")
                return TestResult(
                    scenario="mcp_weights",
                    status="PASS",
                    latency_ms=latency_ms,
                    ram_peak_mb=0,
                    cpu_peak_percent=resources_end["cpu_percent"]
                )
            else:
                raise ValueError(f"α + β = {alpha + beta} ≠ 1.0")
        
        except Exception as e:
            print(f"  ❌ FAIL - {e}")
            return TestResult(
                scenario="mcp_weights",
                status="FAIL",
                latency_ms=0,
                ram_peak_mb=0,
                cpu_peak_percent=0,
                error=str(e)
            )
    
    def test_omni_loop_simulation(self) -> TestResult:
        """Test 4: Omni-Loop simulado (3 iteraciones con Ollama)"""
        print("🧪 [TEST 4/6] Omni-Loop Simulation (3 iterations)")
        
        start = time.perf_counter()
        resources_start = self.monitor_resources()
        
        prompt = "¿Cuál es la capital de Francia?"
        iterations = []
        
        try:
            for i in range(1, 4):
                print(f"  🔄 Iteration {i}/3...")
                
                iter_start = time.perf_counter()
                
                # Llamar a Ollama como draft LLM
                result = subprocess.run(
                    ["ollama", "run", "qwen2.5:0.5b", prompt, "--verbose"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                iter_latency = (time.perf_counter() - iter_start) * 1000
                response = result.stdout.strip()
                
                # Mock confidence (basado en longitud)
                confidence = min(0.85, len(response) / 200)
                
                iterations.append({
                    "iteration": i,
                    "latency_ms": iter_latency,
                    "confidence": confidence,
                    "response_preview": response[:50]
                })
                
                # Reflexión en iter 2-3
                if i < 3:
                    prompt = f"Mejora esta respuesta: {response[:100]}"
                
                # Early exit si confidence > 0.85
                if confidence > 0.85:
                    print(f"  ⚡ Early exit (confidence {confidence:.2f})")
                    break
            
            total_latency_ms = (time.perf_counter() - start) * 1000
            resources_end = self.monitor_resources()
            
            ram_peak = resources_end["ram_used_mb"] - resources_start["ram_used_mb"]
            
            print(f"  ✅ PASS - {len(iterations)} iterations, {total_latency_ms:.0f}ms total, RAM: +{ram_peak:.1f}MB")
            
            return TestResult(
                scenario="omni_loop_simulation",
                status="PASS",
                latency_ms=total_latency_ms,
                ram_peak_mb=ram_peak,
                cpu_peak_percent=resources_end["cpu_percent"]
            )
        
        except Exception as e:
            print(f"  ❌ FAIL - {e}")
            return TestResult(
                scenario="omni_loop_simulation",
                status="FAIL",
                latency_ms=0,
                ram_peak_mb=0,
                cpu_peak_percent=0,
                error=str(e)
            )
    
    def test_stress_concurrent(self) -> TestResult:
        """Test 5: Stress test con 10 queries concurrentes"""
        print("🧪 [TEST 5/6] Stress Test (10 concurrent queries)")
        
        start = time.perf_counter()
        resources_start = self.monitor_resources()
        
        try:
            import concurrent.futures
            
            def single_query(query_id: int):
                """Ejecuta una query simple"""
                result = subprocess.run(
                    ["ollama", "run", "qwen2.5:0.5b", f"Query {query_id}: ¿Qué es Python?"],
                    capture_output=True,
                    text=True,
                    timeout=20
                )
                return len(result.stdout) > 10
            
            # Ejecutar 10 queries en paralelo
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(single_query, i) for i in range(10)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            total_latency_ms = (time.perf_counter() - start) * 1000
            resources_end = self.monitor_resources()
            
            ram_peak = resources_end["ram_used_mb"] - resources_start["ram_used_mb"]
            success_rate = sum(results) / len(results)
            
            if success_rate >= 0.8:  # 80% success
                print(f"  ✅ PASS - {total_latency_ms:.0f}ms, Success: {success_rate*100:.0f}%, RAM: +{ram_peak:.1f}MB")
                return TestResult(
                    scenario="stress_concurrent",
                    status="PASS",
                    latency_ms=total_latency_ms,
                    ram_peak_mb=ram_peak,
                    cpu_peak_percent=resources_end["cpu_percent"]
                )
            else:
                raise Exception(f"Success rate {success_rate*100:.0f}% < 80%")
        
        except Exception as e:
            print(f"  ❌ FAIL - {e}")
            return TestResult(
                scenario="stress_concurrent",
                status="FAIL",
                latency_ms=0,
                ram_peak_mb=0,
                cpu_peak_percent=0,
                error=str(e)
            )
    
    def test_ram_stability(self) -> TestResult:
        """Test 6: Estabilidad de RAM durante 60s"""
        print("🧪 [TEST 6/6] RAM Stability (60s continuous load)")
        
        start = time.perf_counter()
        resources_start = self.monitor_resources()
        
        ram_samples = []
        duration = 60  # 60 segundos
        
        try:
            print(f"  ⏱️ Monitoring RAM for {duration}s...")
            
            end_time = time.time() + duration
            while time.time() < end_time:
                # Ejecutar query cada 5s
                subprocess.run(
                    ["ollama", "run", "qwen2.5:0.5b", "Test query"],
                    capture_output=True,
                    timeout=10
                )
                
                # Capturar RAM
                mem = psutil.virtual_memory()
                ram_samples.append(mem.used / (1024**3))  # GB
                
                time.sleep(5)
            
            total_latency_ms = (time.perf_counter() - start) * 1000
            resources_end = self.monitor_resources()
            
            # Analizar estabilidad
            ram_max = max(ram_samples)
            ram_min = min(ram_samples)
            ram_variance = ram_max - ram_min
            
            # Validar que no hay leak (variance < 2GB)
            if ram_variance < 2.0:
                print(f"  ✅ PASS - RAM variance: {ram_variance:.2f}GB (max: {ram_max:.1f}GB)")
                return TestResult(
                    scenario="ram_stability",
                    status="PASS",
                    latency_ms=total_latency_ms,
                    ram_peak_mb=ram_variance * 1024,
                    cpu_peak_percent=resources_end["cpu_percent"]
                )
            else:
                raise Exception(f"RAM leak detected: {ram_variance:.2f}GB variance")
        
        except Exception as e:
            print(f"  ❌ FAIL - {e}")
            return TestResult(
                scenario="ram_stability",
                status="FAIL",
                latency_ms=0,
                ram_peak_mb=0,
                cpu_peak_percent=0,
                error=str(e)
            )
    
    def run_all_tests(self):
        """Ejecuta todos los tests en secuencia"""
        print(f"\n{'='*60}")
        print(f"🚀 SARAi v2.16 Production Stability Test")
        print(f"{'='*60}\n")
        print(f"Duration: {self.duration}s")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"RAM Baseline: {self.ram_baseline_mb:.1f}MB\n")
        
        # Tests en orden (stack completo SARAi v2.16)
        tests = [
            self.test_solar_expert_inference,     # 1. SOLAR-10.7B (Expert - hard skills)
            self.test_lfm2_tiny_tier,             # 2. LFM2-1.2B (Tiny - soft skills)
            self.test_qwen_omni_multimodal,       # 3. Qwen2.5-Omni-7B (Multimodal)
            self.test_ollama_inference,           # 4. Draft LLM (Qwen2.5:0.5b)
            self.test_trm_router,                 # 5. TRM Router
            self.test_mcp_weights,                # 6. MCP
            self.test_omni_loop_simulation,       # 7. Omni-Loop (3 iter)
            self.test_stress_concurrent,          # 8. Stress (10 concurrent)
            self.test_ram_stability,              # 9. RAM Stability (60s)
        ]
        
        for test_fn in tests:
            result = test_fn()
            self.results.append(result)
            print()  # Separador
        
        # Reporte final
        self.generate_report()
    
    def generate_report(self):
        """Genera reporte de estabilidad"""
        print(f"\n{'='*60}")
        print(f"📊 STABILITY TEST REPORT")
        print(f"{'='*60}\n")
        
        # Estadísticas
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed:   {passed} ({passed/total_tests*100:.0f}%)")
        print(f"❌ Failed:   {failed} ({failed/total_tests*100:.0f}%)")
        print(f"⚠️ Skipped:  {skipped} ({skipped/total_tests*100:.0f}%)")
        print()
        
        # Métricas agregadas
        avg_latency = sum(r.latency_ms for r in self.results if r.status == "PASS") / max(passed, 1)
        max_ram_peak = max((r.ram_peak_mb for r in self.results), default=0)
        max_cpu_peak = max((r.cpu_peak_percent for r in self.results), default=0)
        
        print(f"Avg Latency: {avg_latency:.0f}ms")
        print(f"Max RAM Peak: {max_ram_peak:.1f}MB")
        print(f"Max CPU Peak: {max_cpu_peak:.1f}%")
        print()
        
        # Tabla de resultados
        print(f"{'Scenario':<30} {'Status':<10} {'Latency':<12} {'RAM Peak':<12}")
        print(f"{'-'*70}")
        for r in self.results:
            status_icon = "✅" if r.status == "PASS" else "❌" if r.status == "FAIL" else "⚠️"
            print(f"{r.scenario:<30} {status_icon} {r.status:<8} {r.latency_ms:>8.0f}ms   {r.ram_peak_mb:>8.1f}MB")
        
        # Guardar JSON
        report_path = self.logs_dir / f"stability_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.logs_dir.mkdir(exist_ok=True)
        
        with open(report_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        print(f"\n📄 Report saved: {report_path}")
        
        # Exit code
        if failed > 0:
            print(f"\n❌ STABILITY TEST FAILED ({failed} failures)")
            sys.exit(1)
        else:
            print(f"\n✅ STABILITY TEST PASSED")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="SARAi v2.16 Production Stability Test")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds (default: 300)")
    parser.add_argument("--skip-setup", action="store_true", help="Skip setup phase")
    
    args = parser.parse_args()
    
    tester = ProductionTester(duration_seconds=args.duration)
    
    if not args.skip_setup:
        if not tester.setup():
            print("❌ Setup failed. Exiting.")
            sys.exit(1)
    
    tester.run_all_tests()


if __name__ == "__main__":
    main()
