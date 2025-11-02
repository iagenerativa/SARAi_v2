#!/usr/bin/env python3
"""
SARAi v2.14 - Script de Validación Rápida
==========================================

Ejecuta las verificaciones más importantes del checklist de auditoría.
Útil para CI/CD, validación post-deployment y troubleshooting.

Uso:
    python scripts/quick_validate.py
    python scripts/quick_validate.py --verbose
    python scripts/quick_validate.py --section config
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_result(passed: bool, message: str, details: str = ""):
    """Print test result"""
    icon = f"{Colors.GREEN}✅{Colors.RESET}" if passed else f"{Colors.RED}❌{Colors.RESET}"
    print(f"{icon} {message}")
    if details and not passed:
        print(f"   {Colors.YELLOW}{details}{Colors.RESET}")


def run_command(cmd: str, capture: bool = True) -> Tuple[bool, str]:
    """Run shell command and return (success, output)"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout if capture else ""
    except subprocess.TimeoutExpired:
        return False, "Command timeout"
    except Exception as e:
        return False, str(e)


def validate_config() -> Dict[str, bool]:
    """Sección 1: Configuración Base"""
    print_header("1. Configuración Base")
    results = {}
    
    # Verificar variables de entorno críticas
    env_vars = {
        'OLLAMA_BASE_URL': os.getenv('OLLAMA_BASE_URL'),
        'SOLAR_MODEL_NAME': os.getenv('SOLAR_MODEL_NAME'),
        'HOME_ASSISTANT_URL': os.getenv('HOME_ASSISTANT_URL'),
    }
    
    for var, value in env_vars.items():
        passed = value is not None
        print_result(passed, f"Variable {var}", f"No definida" if not passed else "")
        results[f"env_{var}"] = passed
    
    # Verificar modelos configurados
    try:
        from core.unified_model_wrapper import ModelRegistry
        registry = ModelRegistry()
        registry.load_config()
        model_count = len(registry._config)
        passed = model_count >= 8
        print_result(passed, f"Modelos configurados: {model_count}", 
                    "Menos de 8 modelos" if not passed else "")
        results['model_count'] = passed
    except Exception as e:
        print_result(False, "Carga de configuración", str(e))
        results['model_count'] = False
    
    # Verificar sin IPs hardcodeadas
    success, _ = run_command('grep -r "192\\.168" config/ 2>/dev/null')
    passed = not success  # grep retorna 0 si encuentra, queremos que NO encuentre
    print_result(passed, "Sin IPs hardcodeadas en config/")
    results['no_hardcoded_ips'] = passed
    
    return results


def validate_health() -> Dict[str, bool]:
    """Sección 2: Health Endpoints"""
    print_header("2. Health Endpoints")
    results = {}
    
    # /health endpoint
    success, output = run_command('curl -s http://localhost:8080/health 2>/dev/null')
    passed = success and 'HEALTHY' in output
    print_result(passed, "/health endpoint", "No responde o no HEALTHY" if not passed else "")
    results['health_html'] = passed
    
    # /health JSON
    success, output = run_command(
        'curl -s -H "Accept: application/json" http://localhost:8080/health 2>/dev/null'
    )
    passed = success and '"status"' in output
    print_result(passed, "/health JSON", "No responde o formato incorrecto" if not passed else "")
    results['health_json'] = passed
    
    # /metrics endpoint
    success, output = run_command('curl -s http://localhost:8080/metrics 2>/dev/null')
    passed = success and 'sarai_' in output
    print_result(passed, "/metrics Prometheus", "No expone métricas" if not passed else "")
    results['metrics'] = passed
    
    return results


def validate_tests() -> Dict[str, bool]:
    """Sección 3: Tests"""
    print_header("3. Tests Unitarios e Integración")
    results = {}
    
    # Tests del wrapper
    success, output = run_command('pytest tests/test_unified_wrapper.py -v --tb=short -q 2>&1')
    passed = success and 'passed' in output.lower()
    test_count = output.count(' PASSED') if passed else 0
    print_result(passed, f"Unified Wrapper tests: {test_count}", 
                "Tests fallaron" if not passed else "")
    results['wrapper_tests'] = passed
    
    return results


def validate_logs() -> Dict[str, bool]:
    """Sección 4: Auditoría de Logs"""
    print_header("4. Auditoría de Logs (HMAC + SHA-256)")
    results = {}
    
    # 4.1 Logs estructurados
    log_files = list(Path('logs').glob('*.jsonl')) if Path('logs').exists() else []
    passed = len(log_files) > 0
    print_result(passed, f"Logs estructurados: {len(log_files)} archivos", 
                "No hay logs JSONL" if not passed else "")
    results['log_structure'] = passed
    
    # 4.2 Sidecars de verificación
    sidecar_count = len(list(Path('logs').glob('*.sha256'))) + len(list(Path('logs').glob('*.hmac')))
    passed = sidecar_count > 0
    print_result(passed, f"Sidecars de verificación: {sidecar_count}", 
                "Sin archivos .sha256 o .hmac" if not passed else "")
    results['log_sidecars'] = passed
    
    # 4.3 Verificación criptográfica (NEW)
    if sidecar_count > 0:
        try:
            # Ejecutar script de verificación
            result = subprocess.run(
                ['python3', 'scripts/verify_all_logs.py'],
                capture_output=True,
                text=True,
                timeout=60
            )
            passed = result.returncode == 0
            print_result(passed, "Verificación criptográfica de logs",
                        "Algunos logs tienen firmas inválidas" if not passed else "")
            results['log_crypto_verification'] = passed
        except subprocess.TimeoutExpired:
            print_result(False, "Verificación criptográfica timeout")
            results['log_crypto_verification'] = False
        except Exception as e:
            print_result(False, f"Verificación criptográfica error: {e}")
            results['log_crypto_verification'] = False
    else:
        print_result(True, "Verificación criptográfica: N/A (sin logs)")
        results['log_crypto_verification'] = True  # No es un fallo
    
    return results


def validate_docker() -> Dict[str, bool]:
    """Sección 6: Hardening Docker"""
    print_header("6. Hardening de Contenedores")
    results = {}
    
    # Verificar no-new-privileges
    success, output = run_command(
        'docker inspect sarai-omni-engine 2>/dev/null | jq ".[0].HostConfig.SecurityOpt" 2>/dev/null'
    )
    passed = success and 'no-new-privileges:true' in output
    print_result(passed, "no-new-privileges", "No configurado" if not passed else "")
    results['no_new_privileges'] = passed
    
    # Verificar cap_drop ALL
    success, output = run_command(
        'docker inspect sarai-omni-engine 2>/dev/null | jq ".[0].HostConfig.CapDrop" 2>/dev/null'
    )
    passed = success and 'ALL' in output
    print_result(passed, "cap_drop ALL", "No configurado" if not passed else "")
    results['cap_drop'] = passed
    
    # Verificar read-only
    success, output = run_command(
        'docker inspect sarai-omni-engine 2>/dev/null | jq ".[0].HostConfig.ReadonlyRootfs" 2>/dev/null'
    )
    passed = success and 'true' in output
    print_result(passed, "read-only filesystem", "No configurado" if not passed else "")
    results['read_only'] = passed
    
    return results


def validate_skills() -> Dict[str, bool]:
    """Sección 8: Skills Phoenix"""
    print_header("8. Skills Phoenix (Detección)")
    results = {}
    
    try:
        from core.mcp import detect_and_apply_skill
        
        test_cases = [
            ('Cómo crear una función en Python', 'programming'),
            ('Analizar error de base de datos', 'diagnosis'),
            ('Estrategia de inversión ROI', 'financial'),
            ('Escribe una historia corta', 'creative'),
        ]
        
        passed_count = 0
        for query, expected in test_cases:
            skill = detect_and_apply_skill(query, 'solar')
            detected = skill['name'] if skill else None
            passed = detected == expected
            if passed:
                passed_count += 1
            print_result(passed, f"{query[:35]}... → {detected}")
        
        overall_passed = passed_count == len(test_cases)
        results['skill_detection'] = overall_passed
        
    except Exception as e:
        print_result(False, "Detección de skills", str(e))
        results['skill_detection'] = False
    
    return results


def validate_layers() -> Dict[str, bool]:
    """Sección 9: Layers Architecture"""
    print_header("9. Layers Architecture (Estado)")
    results = {}
    
    # Layer 2: Tone memory
    tone_memory_path = Path('state/layer2_tone_memory.jsonl')
    if tone_memory_path.exists():
        entries = sum(1 for _ in open(tone_memory_path))
        passed = entries <= 256
        print_result(passed, f"Tone memory: {entries} entradas (max 256)", 
                    "Buffer excede límite" if not passed else "")
        results['tone_memory'] = passed
    else:
        print_result(False, "Tone memory", "Archivo no existe")
        results['tone_memory'] = False
    
    # Layer 3: Tone bridge
    try:
        from core.layer3_fluidity.tone_bridge import get_tone_bridge
        bridge = get_tone_bridge()
        profile = bridge.update('happy', 0.8, 0.7)
        passed = profile.style == 'energetic_positive'
        print_result(passed, f"Estilo inferido: {profile.style}")
        results['tone_bridge'] = passed
    except Exception as e:
        print_result(False, "Tone bridge", str(e))
        results['tone_bridge'] = False
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validación rápida de SARAi v2.14')
    parser.add_argument('--verbose', '-v', action='store_true', help='Modo verbose')
    parser.add_argument('--section', '-s', help='Ejecutar solo una sección específica',
                       choices=['config', 'health', 'tests', 'logs', 'docker', 'skills', 'layers'])
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}SARAi v2.14 - Validación Rápida{Colors.RESET}")
    print(f"{'='*60}\n")
    
    all_results = {}
    
    sections = {
        'config': validate_config,
        'health': validate_health,
        'tests': validate_tests,
        'logs': validate_logs,
        'docker': validate_docker,
        'skills': validate_skills,
        'layers': validate_layers,
    }
    
    # Si se especifica sección, solo ejecutar esa
    if args.section:
        sections = {args.section: sections[args.section]}
    
    # Ejecutar validaciones
    for name, func in sections.items():
        try:
            results = func()
            all_results.update(results)
        except Exception as e:
            print(f"{Colors.RED}Error en sección {name}: {e}{Colors.RESET}")
    
    # Resumen final
    print_header("Resumen")
    total = len(all_results)
    passed = sum(1 for v in all_results.values() if v)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"Total: {passed}/{total} verificaciones pasadas ({percentage:.1f}%)")
    
    if percentage >= 95:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ VALIDACIÓN APROBADA{Colors.RESET}")
        return 0
    elif percentage >= 80:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  VALIDACIÓN APROBADA CON OBSERVACIONES{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ VALIDACIÓN RECHAZADA{Colors.RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
