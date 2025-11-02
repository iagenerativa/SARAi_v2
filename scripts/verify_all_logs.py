#!/usr/bin/env python3
"""
SARAi v2.14 - Verificador de Integridad de Logs
================================================

Verifica la integridad criptográfica de todos los logs del sistema:
- Logs de voz (HMAC)
- Logs web (SHA-256)
- Logs de feedback (SHA-256)

Uso:
    python scripts/verify_all_logs.py
    python scripts/verify_all_logs.py --date 2025-01-01
    python scripts/verify_all_logs.py --verbose
"""

import os
import sys
import json
import hashlib
import hmac
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


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
    """Print verification result"""
    if passed:
        icon = f"{Colors.GREEN}✅{Colors.RESET}"
        print(f"{icon} {message}")
    else:
        icon = f"{Colors.RED}❌{Colors.RESET}"
        print(f"{icon} {message}")
        if details:
            print(f"   {Colors.YELLOW}→ {details}{Colors.RESET}")


def verify_sha256_log(log_path: Path, hash_path: Path, verbose: bool = False) -> Tuple[bool, str]:
    """
    Verifica integridad de un log con SHA-256 sidecar
    
    Returns:
        (success, error_message)
    """
    if not log_path.exists():
        return False, f"Log no existe: {log_path}"
    
    if not hash_path.exists():
        return False, f"Sidecar SHA-256 no existe: {hash_path}"
    
    try:
        with open(log_path, 'r') as f_log, open(hash_path, 'r') as f_hash:
            for line_num, (log_line, expected_hash) in enumerate(zip(f_log, f_hash), 1):
                # Calcular hash de la línea (sin newline final)
                computed_hash = hashlib.sha256(log_line.strip().encode('utf-8')).hexdigest()
                expected_hash = expected_hash.strip()
                
                if computed_hash != expected_hash:
                    return False, f"Línea {line_num}: Hash mismatch (esperado: {expected_hash[:16]}..., obtenido: {computed_hash[:16]}...)"
                
                if verbose and line_num % 100 == 0:
                    print(f"  Verificadas {line_num} líneas...")
        
        return True, ""
    
    except Exception as e:
        return False, f"Error durante verificación: {str(e)}"


def verify_hmac_log(log_path: Path, hmac_path: Path, secret_key: bytes, verbose: bool = False) -> Tuple[bool, str]:
    """
    Verifica integridad de un log con HMAC sidecar
    
    Returns:
        (success, error_message)
    """
    if not log_path.exists():
        return False, f"Log no existe: {log_path}"
    
    if not hmac_path.exists():
        return False, f"Sidecar HMAC no existe: {hmac_path}"
    
    try:
        with open(log_path, 'r') as f_log, open(hmac_path, 'r') as f_hmac:
            for line_num, (log_line, expected_hmac) in enumerate(zip(f_log, f_hmac), 1):
                # Parsear JSON de la línea
                try:
                    entry = json.loads(log_line.strip())
                except json.JSONDecodeError:
                    return False, f"Línea {line_num}: JSON inválido"
                
                # Recalcular HMAC
                entry_str = json.dumps(entry, ensure_ascii=False, sort_keys=True)
                computed_hmac = hmac.new(secret_key, entry_str.encode(), hashlib.sha256).hexdigest()
                expected_hmac = expected_hmac.strip()
                
                if computed_hmac != expected_hmac:
                    return False, f"Línea {line_num}: HMAC mismatch (esperado: {expected_hmac[:16]}..., obtenido: {computed_hmac[:16]}...)"
                
                if verbose and line_num % 100 == 0:
                    print(f"  Verificadas {line_num} líneas...")
        
        return True, ""
    
    except Exception as e:
        return False, f"Error durante verificación: {str(e)}"


def verify_all_logs(date_filter: str = None, verbose: bool = False) -> Dict[str, bool]:
    """
    Verifica todos los logs del sistema
    
    Args:
        date_filter: Fecha en formato YYYY-MM-DD (None = todos)
        verbose: Modo verbose
    
    Returns:
        Dict con resultados por tipo de log
    """
    logs_dir = Path('logs')
    results = {}
    
    # Obtener secret key para HMAC
    secret_key = os.getenv('HMAC_SECRET_KEY', 'default-secret').encode()
    
    # 1. Logs de voz (HMAC)
    print_header("1. LOGS DE VOZ (HMAC)")
    
    if date_filter:
        voice_logs = [logs_dir / f"voice_interactions_{date_filter}.jsonl"]
    else:
        voice_logs = sorted(logs_dir.glob("voice_interactions_*.jsonl"))
    
    voice_passed = 0
    voice_total = 0
    
    for log_file in voice_logs:
        if not log_file.exists():
            continue
        
        voice_total += 1
        hmac_file = log_file.with_suffix('.jsonl.hmac')
        
        success, error = verify_hmac_log(log_file, hmac_file, secret_key, verbose)
        
        if success:
            voice_passed += 1
            print_result(True, f"{log_file.name}")
        else:
            print_result(False, f"{log_file.name}", error)
    
    results['voice'] = voice_passed == voice_total and voice_total > 0
    
    if voice_total == 0:
        print_result(True, "Sin logs de voz (sistema recién instalado)")
        results['voice'] = True  # No es un fallo
    
    # 2. Logs web (SHA-256)
    print_header("2. LOGS WEB (SHA-256)")
    
    if date_filter:
        web_logs = [logs_dir / f"web_queries_{date_filter}.jsonl"]
    else:
        web_logs = sorted(logs_dir.glob("web_queries_*.jsonl"))
    
    web_passed = 0
    web_total = 0
    
    for log_file in web_logs:
        if not log_file.exists():
            continue
        
        web_total += 1
        sha_file = log_file.with_suffix('.jsonl.sha256')
        
        success, error = verify_sha256_log(log_file, sha_file, verbose)
        
        if success:
            web_passed += 1
            print_result(True, f"{log_file.name}")
        else:
            print_result(False, f"{log_file.name}", error)
    
    results['web'] = web_passed == web_total and web_total > 0
    
    if web_total == 0:
        print_result(True, "Sin logs web (sistema recién instalado)")
        results['web'] = True  # No es un fallo
    
    # 3. Logs de feedback (SHA-256)
    print_header("3. LOGS DE FEEDBACK (SHA-256)")
    
    feedback_log = logs_dir / "feedback_log.jsonl"
    feedback_sha = logs_dir / "feedback_log.jsonl.sha256"
    
    if feedback_log.exists():
        success, error = verify_sha256_log(feedback_log, feedback_sha, verbose)
        
        if success:
            print_result(True, "feedback_log.jsonl")
            results['feedback'] = True
        else:
            print_result(False, "feedback_log.jsonl", error)
            results['feedback'] = False
    else:
        print_result(True, "Sin logs de feedback (sistema recién instalado)")
        results['feedback'] = True  # No es un fallo
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Verificar integridad de logs SARAi v2.14')
    parser.add_argument('--date', '-d', help='Fecha específica (YYYY-MM-DD)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Modo verbose')
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}SARAi v2.14 - Verificación de Integridad de Logs{Colors.RESET}")
    print(f"{'='*60}\n")
    
    if args.date:
        print(f"Filtrando por fecha: {args.date}\n")
    
    results = verify_all_logs(args.date, args.verbose)
    
    # Resumen final
    print_header("RESUMEN")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"Tipos de logs verificados: {passed}/{total}")
    print(f"  • Logs de voz (HMAC):    {'✅' if results.get('voice') else '❌'}")
    print(f"  • Logs web (SHA-256):    {'✅' if results.get('web') else '❌'}")
    print(f"  • Logs feedback (SHA-256): {'✅' if results.get('feedback') else '❌'}")
    
    print()
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ VERIFICACIÓN APROBADA{Colors.RESET}")
        print(f"\nTodos los logs tienen integridad criptográfica verificada.")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ VERIFICACIÓN RECHAZADA{Colors.RESET}")
        print(f"\nSe detectaron logs corruptos o con firmas inválidas.")
        print(f"Considera activar Safe Mode hasta investigar la causa.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
