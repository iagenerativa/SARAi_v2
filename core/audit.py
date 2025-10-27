"""
SARAi v2.9 - Audit Module con Sentinel Mode
Sistema de auditoría inmutable con modo de autoprotección

NEW v2.9:
- Verificación SHA-256 de logs
- Modo Seguro global (bloquea reentrenamiento si logs corruptos)
- Webhook de notificación crítica
- Sistema de cuarentena para logs sospechosos
"""

import hashlib
import threading
import time
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests

# ---------- GLOBAL SENTINEL MODE ----------
# Flag global compartido por todos los módulos
GLOBAL_SAFE_MODE = threading.Event()
_safe_mode_lock = threading.RLock()
_safe_mode_reason: Optional[str] = None


def is_safe_mode() -> bool:
    """Retorna True si el sistema está en Modo Seguro."""
    return GLOBAL_SAFE_MODE.is_set()


def get_safe_mode_reason() -> Optional[str]:
    """Obtiene la razón por la que se activó el Modo Seguro."""
    with _safe_mode_lock:
        return _safe_mode_reason


def activate_safe_mode(reason: str):
    """
    Activa el Modo Seguro globalmente.
    
    Efectos:
    - Bloquea reentrenamiento del MCP
    - Bloquea carga de nuevos skills
    - Sistema solo responde con modelos actuales
    - Notifica vía webhook
    """
    global _safe_mode_reason
    
    with _safe_mode_lock:
        if not GLOBAL_SAFE_MODE.is_set():
            _safe_mode_reason = reason
            GLOBAL_SAFE_MODE.set()
            
            print("=" * 70)
            print("🚨 MODO SEGURO ACTIVADO 🚨")
            print("=" * 70)
            print(f"Razón: {reason}")
            print("El sistema continuará operando pero:")
            print("  • NO se reentrenará el MCP")
            print("  • NO se cargarán nuevos skills")
            print("  • Solo modelos verificados en uso")
            print("=" * 70)
            
            # Enviar notificación crítica
            send_critical_webhook(reason)


def deactivate_safe_mode():
    """
    Desactiva el Modo Seguro (solo si la causa fue resuelta).
    
    ADVERTENCIA: Solo llamar después de verificar que el problema
    que causó el Modo Seguro ha sido completamente resuelto.
    """
    global _safe_mode_reason
    
    with _safe_mode_lock:
        if GLOBAL_SAFE_MODE.is_set():
            print("✅ Modo Seguro desactivado - Sistema vuelve a operación normal")
            GLOBAL_SAFE_MODE.clear()
            _safe_mode_reason = None


from contextlib import contextmanager

@contextmanager
def disable_safe_mode_temp():
    """
    Context manager para deshabilitar temporalmente el Modo Seguro.
    Útil para tests que necesitan acceso a funcionalidades bloqueadas.
    
    Uso:
        with disable_safe_mode_temp():
            # Código que requiere Safe Mode OFF
            resultado = funcion_bloqueada()
    """
    global _safe_mode_reason
    
    was_active = GLOBAL_SAFE_MODE.is_set()
    previous_reason = _safe_mode_reason
    
    if was_active:
        deactivate_safe_mode()
    
    try:
        yield
    finally:
        if was_active:
            # Restaurar estado previo
            with _safe_mode_lock:
                _safe_mode_reason = previous_reason
                GLOBAL_SAFE_MODE.set()


# ---------- AUDITORÍA DE LOGS ----------

def hash_line(line: str) -> str:
    """Calcula SHA-256 de una línea de log."""
    return hashlib.sha256(line.strip().encode('utf-8')).hexdigest()


def verify_log_file(log_path: Path, hash_path: Path) -> Tuple[bool, int, int]:
    """
    Verifica integridad de un archivo de log contra su hash sidecar.
    
    Returns:
        (is_valid, total_lines, corrupted_lines)
    """
    if not log_path.exists():
        return (False, 0, 0)
    
    if not hash_path.exists():
        # Log sin hash = no verificable pero no corrupto
        return (True, 0, 0)
    
    corrupted = 0
    total = 0
    
    try:
        with log_path.open() as f_log, hash_path.open() as f_hash:
            for line, expected_hash in zip(f_log, f_hash):
                total += 1
                computed_hash = hash_line(line)
                
                if computed_hash != expected_hash.strip():
                    corrupted += 1
                    print(f"⚠️ Línea {total} corrupta en {log_path.name}")
        
        is_valid = (corrupted == 0)
        return (is_valid, total, corrupted)
        
    except Exception as e:
        print(f"❌ Error verificando {log_path}: {e}")
        return (False, 0, 0)


def verify_all_logs(logs_dir: Path = Path("logs")) -> Dict[str, any]:
    """
    Verifica integridad de todos los logs en el directorio.
    
    Returns:
        {
            "total_files": int,
            "verified": int,
            "corrupted": int,
            "details": [...]
        }
    """
    if not logs_dir.exists():
        return {"total_files": 0, "verified": 0, "corrupted": 0, "details": []}
    
    results = {
        "total_files": 0,
        "verified": 0,
        "corrupted": 0,
        "details": []
    }
    
    for log_file in sorted(logs_dir.glob("*.jsonl")):
        hash_file = log_file.with_suffix(".jsonl.sha256")
        
        is_valid, total_lines, corrupted_lines = verify_log_file(log_file, hash_file)
        
        results["total_files"] += 1
        
        if is_valid:
            results["verified"] += 1
        else:
            results["corrupted"] += 1
        
        results["details"].append({
            "file": log_file.name,
            "valid": is_valid,
            "total_lines": total_lines,
            "corrupted_lines": corrupted_lines
        })
    
    return results


def audit_logs_and_activate_safe_mode(logs_dir: Path = Path("logs")):
    """
    NEW v2.9: Audita logs y activa Modo Seguro si detecta corrupción.
    
    Esta función debe ejecutarse ANTES de cada ciclo de online tuning.
    """
    print("🔍 Iniciando auditoría de logs...")
    
    results = verify_all_logs(logs_dir)
    
    print(f"📊 Auditoría completada:")
    print(f"  • Total archivos: {results['total_files']}")
    print(f"  • Verificados: {results['verified']}")
    print(f"  • Corruptos: {results['corrupted']}")
    
    if results['corrupted'] > 0:
        # CRÍTICO: Logs corruptos detectados
        corrupted_files = [
            d['file'] for d in results['details'] if not d['valid']
        ]
        
        reason = (
            f"Auditoría de logs FALLIDA: {results['corrupted']} archivo(s) corrupto(s)\n"
            f"Archivos afectados: {', '.join(corrupted_files)}\n"
            f"El reentrenamiento del MCP está bloqueado hasta resolver la corrupción"
        )
        
        activate_safe_mode(reason)
        
        # Mover logs corruptos a cuarentena
        quarantine_dir = logs_dir / "quarantine"
        quarantine_dir.mkdir(exist_ok=True)
        
        for detail in results['details']:
            if not detail['valid']:
                corrupted_path = logs_dir / detail['file']
                quarantine_path = quarantine_dir / f"{detail['file']}.{int(time.time())}"
                
                try:
                    corrupted_path.rename(quarantine_path)
                    print(f"📦 Movido a cuarentena: {detail['file']}")
                except Exception as e:
                    print(f"❌ Error moviendo a cuarentena: {e}")
        
        return False
    
    else:
        print("✅ Auditoría exitosa - Logs íntegros")
        return True


# ---------- WEBHOOK NOTIFICATIONS ----------

def send_critical_webhook(message: str, webhook_url: Optional[str] = None):
    """
    Envía notificación crítica vía webhook.
    
    Soporta:
    - Slack
    - Discord
    - Generic webhook (JSON POST)
    """
    if webhook_url is None:
        webhook_url = os.getenv("SARAI_WEBHOOK_URL")
    
    if not webhook_url:
        print("⚠️ No hay webhook configurado (SARAI_WEBHOOK_URL)")
        return
    
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "severity": "CRITICAL",
        "system": "SARAi v2.9",
        "event": "SAFE_MODE_ACTIVATED",
        "message": message,
        "hostname": os.uname().nodename
    }
    
    try:
        # Detectar tipo de webhook por URL
        if "slack.com" in webhook_url:
            payload = {"text": f"🚨 SARAi CRITICAL: {message}"}
        elif "discord.com" in webhook_url:
            payload = {"content": f"🚨 **SARAi CRITICAL**\n{message}"}
        
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Webhook enviado exitosamente")
        else:
            print(f"⚠️ Webhook falló: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error enviando webhook: {e}")


# ---------- MONITORING DAEMON ----------

class AuditDaemon:
    """
    Daemon que verifica periódicamente la integridad de los logs.
    
    NEW v2.9: Sistema de vigilancia continua.
    """
    
    def __init__(self, interval_minutes: int = 60):
        self.interval = interval_minutes * 60
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def start(self):
        """Inicia el daemon de auditoría."""
        if self.running:
            print("⚠️ Audit daemon ya está corriendo")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"✅ Audit daemon iniciado (intervalo: {self.interval // 60} min)")
    
    def stop(self):
        """Detiene el daemon de auditoría."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("🛑 Audit daemon detenido")
    
    def _run(self):
        """Loop principal del daemon."""
        while self.running:
            try:
                audit_logs_and_activate_safe_mode()
            except Exception as e:
                print(f"❌ Error en audit daemon: {e}")
            
            # Esperar intervalo
            time.sleep(self.interval)


# ---------- CLI ----------

def main():
    """CLI para auditoría manual."""
    import sys
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="SARAi v2.9 - Audit Tool")
    parser.add_argument("--verify", action="store_true", help="Verificar integridad de logs")
    parser.add_argument("--daemon", action="store_true", help="Iniciar daemon de auditoría")
    parser.add_argument("--deactivate-safe-mode", action="store_true", help="Desactivar Modo Seguro")
    parser.add_argument("--logs-dir", default="logs", help="Directorio de logs")
    
    args = parser.parse_args()
    
    if args.verify:
        audit_logs_and_activate_safe_mode(Path(args.logs_dir))
    
    elif args.daemon:
        daemon = AuditDaemon(interval_minutes=60)
        daemon.start()
        
        print("Daemon corriendo. Presiona Ctrl+C para detener...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            daemon.stop()
    
    elif args.deactivate_safe_mode:
        if is_safe_mode():
            print(f"Razón actual: {get_safe_mode_reason()}")
            confirm = input("¿Seguro que quieres desactivar Modo Seguro? [y/N]: ")
            if confirm.lower() == 'y':
                deactivate_safe_mode()
        else:
            print("ℹ️ El sistema no está en Modo Seguro")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
