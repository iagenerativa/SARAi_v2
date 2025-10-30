#!/usr/bin/env python3
"""
SARAi v2.16.3 - Voice Pipeline Benchmark
========================================

Script interactivo para ejecutar y comparar tests de voz:
- Opción A: Pipeline sin LLM (Audio → Audio directo)
- Opción B: Pipeline completo con LLM (Conversación)

Genera informe de latencias y recomendaciones.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Colores para terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Imprime encabezado destacado"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(text: str):
    """Mensaje de éxito"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Mensaje de advertencia"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_error(text: str):
    """Mensaje de error"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_info(text: str):
    """Mensaje informativo"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


def check_prerequisites() -> Dict[str, bool]:
    """
    Verifica que todos los archivos necesarios existan
    
    Returns:
        Dict con estado de cada prerequisito
    """
    print_header("VERIFICANDO PREREQUISITOS")
    
    base_path = Path(__file__).parent.parent
    
    checks = {
        "audio_encoder_int8": base_path / "models/onnx/audio_encoder_int8.pt",
        "projection_onnx": base_path / "models/onnx/projection.onnx",
        "talker_onnx": base_path / "models/onnx/old/qwen25_audio.onnx",
        "talker_data": base_path / "models/onnx/old/qwen25_audio.onnx.data",
        "token2wav_int8": base_path / "models/onnx/token2wav_int8.pt",
        "lfm2_model": base_path / "models/lfm2/LFM2-1.2B-Q4_K_M.gguf",
    }
    
    results = {}
    for name, path in checks.items():
        exists = path.exists()
        results[name] = exists
        
        if exists:
            size_mb = path.stat().st_size / (1024**2)
            print_success(f"{name}: {path.name} ({size_mb:.1f} MB)")
        else:
            print_error(f"{name}: NO ENCONTRADO - {path}")
    
    # Verificar librerías Python
    print("\n📚 Verificando dependencias Python:")
    
    libraries = {
        "torch": "PyTorch",
        "onnxruntime": "ONNX Runtime",
        "numpy": "NumPy",
        "soundfile": "SoundFile",
        "pyaudio": "PyAudio (grabación)",
    }
    
    for module_name, display_name in libraries.items():
        try:
            __import__(module_name)
            print_success(f"{display_name}")
            results[f"lib_{module_name}"] = True
        except ImportError:
            print_error(f"{display_name} - pip install {module_name}")
            results[f"lib_{module_name}"] = False
    
    return results


def show_menu() -> str:
    """
    Muestra menú interactivo y retorna opción elegida
    
    Returns:
        'A', 'B', o 'Q' (quit)
    """
    print_header("BENCHMARK DE VOZ - SELECCIÓN DE TEST")
    
    print(f"{Colors.BOLD}Opción A: Pipeline sin LLM (Rápido){Colors.ENDC}")
    print("   📊 Audio → Encoder → Projection → Talker → Token2Wav → Audio")
    print("   ⚡ Latencia esperada: ~200ms")
    print("   🎯 Uso: Conversión de voz, síntesis directa")
    print("   ✅ Ideal para: Validar pipeline básico")
    
    print(f"\n{Colors.BOLD}Opción B: Pipeline con LLM (Completo){Colors.ENDC}")
    print("   📊 Audio → Encoder → Projection → LFM2 → Talker → Token2Wav → Audio")
    print("   ⏱️  Latencia esperada: ~1.2-1.5s")
    print("   🎯 Uso: Conversación inteligente, asistente de voz")
    print("   ✅ Ideal para: Validar experiencia completa")
    
    print(f"\n{Colors.BOLD}Opción T: Test Talker ONNX (Ultra-rápido){Colors.ENDC}")
    print("   📊 Solo qwen25_audio.onnx con datos dummy")
    print("   ⚡ Latencia esperada: ~110ms")
    print("   🎯 Uso: Validar ONNX performance")
    print("   ✅ Ideal para: Baseline rápido")
    
    print(f"\n{Colors.BOLD}Q: Salir{Colors.ENDC}")
    
    while True:
        choice = input(f"\n{Colors.OKCYAN}Selecciona opción [A/B/T/Q]: {Colors.ENDC}").strip().upper()
        if choice in ['A', 'B', 'T', 'Q']:
            return choice
        print_error("Opción inválida. Usa A, B, T o Q")


def run_option_t() -> Dict[str, float]:
    """
    Ejecuta Opción T: Test Talker ONNX solo
    
    Returns:
        Dict con métricas de latencia
    """
    print_header("EJECUTANDO OPCIÓN T: TALKER ONNX")
    
    print_info("Ejecutando tests/test_voice_simple_onnx.py...")
    
    import subprocess
    
    try:
        # Ejecutar el test con timeout de 60s
        result = subprocess.run(
            ["python3", "tests/test_voice_simple_onnx.py"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=60,
            input="\n\n\n"  # Responder automáticamente 3 veces
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print_success("Test completado exitosamente")
            
            # Parsear resultados del output
            lines = result.stdout.split('\n')
            metrics = {}
            
            for line in lines:
                if "Min:" in line:
                    metrics['min_ms'] = float(line.split(':')[1].strip().replace('ms', ''))
                elif "Max:" in line:
                    metrics['max_ms'] = float(line.split(':')[1].strip().replace('ms', ''))
                elif "Promedio:" in line:
                    metrics['avg_ms'] = float(line.split(':')[1].strip().replace('ms', ''))
            
            return metrics
        else:
            print_error(f"Test falló con código {result.returncode}")
            if result.stderr:
                print(f"\n{Colors.FAIL}Error:{Colors.ENDC}\n{result.stderr}")
            return {}
            
    except subprocess.TimeoutExpired:
        print_error("Test excedió tiempo máximo (60s)")
        return {}
    except Exception as e:
        print_error(f"Error ejecutando test: {e}")
        return {}


def run_option_a() -> Dict[str, float]:
    """
    Ejecuta Opción A: Pipeline sin LLM
    
    Returns:
        Dict con métricas de latencia
    """
    print_header("EJECUTANDO OPCIÓN A: PIPELINE SIN LLM")
    
    print_warning("Esta opción requiere AutoProcessor de HuggingFace")
    print_info("Primera ejecución descargará ~500MB (una sola vez)")
    
    response = input(f"\n{Colors.OKCYAN}¿Continuar? [s/N]: {Colors.ENDC}").strip().lower()
    
    if response != 's':
        print_info("Operación cancelada")
        return {}
    
    print_info("Ejecutando tests/test_voice_pipeline_completo.py (sin LLM)...")
    
    # TODO: Implementar cuando AutoProcessor esté listo
    print_warning("Implementación pendiente - Requiere AutoProcessor")
    
    return {}


def run_option_b() -> Dict[str, float]:
    """
    Ejecuta Opción B: Pipeline con LLM
    
    Returns:
        Dict con métricas de latencia
    """
    print_header("EJECUTANDO OPCIÓN B: PIPELINE CON LLM")
    
    print_warning("Esta opción requiere:")
    print("   - AutoProcessor de HuggingFace (~500MB)")
    print("   - LFM2-1.2B cargado en memoria (~450MB)")
    
    response = input(f"\n{Colors.OKCYAN}¿Continuar? [s/N]: {Colors.ENDC}").strip().lower()
    
    if response != 's':
        print_info("Operación cancelada")
        return {}
    
    print_info("Ejecutando tests/test_voice_llm_completo.py...")
    
    # TODO: Implementar pipeline con LLM
    print_warning("Implementación pendiente")
    
    return {}


def generate_report(metrics: Dict[str, float], option: str):
    """
    Genera informe de resultados
    
    Args:
        metrics: Métricas capturadas
        option: Opción ejecutada ('A', 'B', o 'T')
    """
    print_header("INFORME DE RESULTADOS")
    
    if not metrics:
        print_warning("No hay métricas para reportar")
        return
    
    print(f"{Colors.BOLD}Test Ejecutado:{Colors.ENDC} Opción {option}")
    
    if option == 'T':
        print(f"\n{Colors.BOLD}Latencias ONNX Talker:{Colors.ENDC}")
        print(f"   Min:      {metrics.get('min_ms', 0):.1f}ms")
        print(f"   Max:      {metrics.get('max_ms', 0):.1f}ms")
        print(f"   Promedio: {metrics.get('avg_ms', 0):.1f}ms")
        
        avg = metrics.get('avg_ms', 0)
        if avg < 120:
            print_success(f"Rendimiento excelente (< 120ms)")
        elif avg < 200:
            print_info(f"Rendimiento bueno (< 200ms)")
        else:
            print_warning(f"Rendimiento por debajo de objetivo (>= 200ms)")
    
    # Guardar resultados en archivo
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = Path(__file__).parent.parent / f"logs/voice_benchmark_{timestamp}.txt"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(f"SARAi Voice Benchmark - Opción {option}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\nMétricas:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")
    
    print_success(f"Informe guardado: {report_file}")


def main():
    """Función principal"""
    
    print_header("SARAi v2.16.3 - Voice Pipeline Benchmark")
    
    # Verificar prerequisitos
    prereqs = check_prerequisites()
    
    critical_missing = [
        k for k, v in prereqs.items() 
        if not v and not k.startswith('lib_')
    ]
    
    if critical_missing:
        print_error(f"\n❌ Faltan archivos críticos: {', '.join(critical_missing)}")
        print_info("Revisa docs/VOICE_TEST_RESULTS.md para más información")
        sys.exit(1)
    
    print_success("\n✅ Todos los prerequisitos cumplidos\n")
    
    # Menú interactivo
    while True:
        choice = show_menu()
        
        if choice == 'Q':
            print_info("Saliendo...")
            break
        
        metrics = {}
        
        if choice == 'T':
            metrics = run_option_t()
        elif choice == 'A':
            metrics = run_option_a()
        elif choice == 'B':
            metrics = run_option_b()
        
        if metrics:
            generate_report(metrics, choice)
        
        print("\n" + "─"*70 + "\n")
        response = input(f"{Colors.OKCYAN}¿Ejecutar otro test? [s/N]: {Colors.ENDC}").strip().lower()
        if response != 's':
            break
    
    print_header("BENCHMARK FINALIZADO")
    print_info("Revisa logs/voice_benchmark_*.txt para resultados completos")
    print_info("Documentación: docs/VOICE_TEST_RESULTS.md")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\n\n⚠️  Benchmark interrumpido por usuario")
        sys.exit(130)
    except Exception as e:
        print_error(f"\n\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
