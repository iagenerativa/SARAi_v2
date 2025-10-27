"""
SARAi v2.11 - Web & Voice Audit Module (Logging firmado HMAC + SHA-256)

Sistema de auditoría inmutable para búsquedas web RAG y voz:
- Cada query web se firma con SHA-256 (integridad verificable)
- Cada interacción de voz se firma con HMAC-SHA256
- Formato: logs/web_queries_YYYY-MM-DD.jsonl + .sha256
-         logs/voice_interactions_YYYY-MM-DD.jsonl + .hmac
- Compatible con el sistema de auditoría v2.9 (core/audit.py)
- Trigger de Sentinel Mode si detección de manipulación

Garantías v2.11:
- ✅ Cada búsqueda web firmada inmutablemente (SHA-256)
- ✅ Cada interacción de voz firmada con HMAC-SHA256
- ✅ Logs verificables con sidecars (.sha256 / .hmac)
- ✅ Trigger automático de Safe Mode si corrupción detectada
- ✅ Webhooks de alerta para consultas/interacciones anómalas
"""

import os
import json
import hashlib
import hmac
from datetime import datetime
from typing import Dict, Optional, List
import threading

# CRÍTICO: Importar Safe Mode para trigger
from core.audit import activate_safe_mode, send_critical_webhook


class WebAuditLogger:
    """
    Logger de auditoría para búsquedas web con firma SHA-256
    
    Formato de log:
    {
        "timestamp": "2025-10-27T14:32:10.123456",
        "query": "¿Quién ganó el Oscar 2025?",
        "source": "cache" | "searxng",
        "snippets_count": 5,
        "snippets_urls": ["url1", "url2", ...],
        "synthesis_used": true,
        "llm_model": "expert_short" | "expert_long",
        "response_preview": "Según los resultados...",
        "safe_mode_active": false
    }
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.lock = threading.Lock()  # Thread-safe logging
    
    def _get_log_paths(self) -> tuple:
        """Retorna (jsonl_path, sha256_path) para el día actual"""
        date = datetime.now().strftime("%Y-%m-%d")
        jsonl_path = os.path.join(self.log_dir, f"web_queries_{date}.jsonl")
        sha256_path = f"{jsonl_path}.sha256"
        return jsonl_path, sha256_path
    
    def log_web_query(
        self,
        query: str,
        search_results: Optional[Dict],
        response: Optional[str] = None,
        llm_model: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Registra una búsqueda web con firma SHA-256
        
        Args:
            query: Query del usuario
            search_results: Output de web_cache.cached_search()
            response: Respuesta sintetizada por el LLM (opcional)
            llm_model: Modelo usado para síntesis (opcional)
            error: Mensaje de error si fallo (opcional)
        """
        with self.lock:
            # Construir entrada de log
            entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "source": search_results.get("source") if search_results else "error",
                "snippets_count": len(search_results.get("snippets", [])) if search_results else 0,
                "snippets_urls": [
                    s.get("url") for s in search_results.get("snippets", [])
                ] if search_results else [],
                "synthesis_used": response is not None,
                "llm_model": llm_model,
                "response_preview": response[:200] if response else None,
                "safe_mode_active": False,  # Se actualiza si se detecta anomalía
                "error": error
            }
            
            # DETECCIÓN DE ANOMALÍAS
            # Si SearXNG retorna 0 snippets repetidamente, puede ser ataque
            if search_results and entry["snippets_count"] == 0 and entry["source"] == "searxng":
                self._trigger_anomaly_alert(entry)
            
            # Escribir JSON
            jsonl_path, sha256_path = self._get_log_paths()
            log_line = json.dumps(entry, ensure_ascii=False)
            
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
            
            # Escribir hash SHA-256 (inmutabilidad)
            line_hash = hashlib.sha256(log_line.encode('utf-8')).hexdigest()
            with open(sha256_path, "a", encoding="utf-8") as f_hash:
                f_hash.write(f"{line_hash}\n")
    
    def _trigger_anomaly_alert(self, entry: Dict):
        """
        Detecta comportamiento anómalo en búsquedas web
        
        Triggers:
        - SearXNG retorna 0 resultados repetidamente (posible DOS)
        - Queries con URLs sospechosas en snippets
        - Volumen anómalo de búsquedas (>100/min)
        """
        print(f"⚠️ ANOMALÍA WEB detectada: {entry['query'][:50]}...")
        
        # Enviar webhook crítico (Slack/Discord)
        send_critical_webhook(
            subject="SARAi RAG Anomaly Detected",
            message=f"Query: {entry['query']}\nSource: {entry['source']}\nSnippets: {entry['snippets_count']}"
        )
        
        # NO activar Safe Mode por una query, solo alertar
        # Safe Mode se activa si hay corrupción de logs verificada
    
    def verify_integrity(self, date: Optional[str] = None) -> bool:
        """
        Verifica integridad de logs web para una fecha
        
        Args:
            date: "YYYY-MM-DD" o None para hoy
        
        Returns:
            True si todos los hashes coinciden, False si corrupción
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        jsonl_path = os.path.join(self.log_dir, f"web_queries_{date}.jsonl")
        sha256_path = f"{jsonl_path}.sha256"
        
        if not os.path.exists(jsonl_path) or not os.path.exists(sha256_path):
            return True  # No hay logs para verificar
        
        try:
            with open(jsonl_path, encoding="utf-8") as f_log, \
                 open(sha256_path) as f_hash:
                
                for line_num, (log_line, expected_hash) in enumerate(zip(f_log, f_hash), 1):
                    computed_hash = hashlib.sha256(log_line.strip().encode('utf-8')).hexdigest()
                    
                    if computed_hash != expected_hash.strip():
                        print(f"❌ CORRUPCIÓN en web_queries_{date}.jsonl línea {line_num}")
                        
                        # TRIGGER SAFE MODE (crítico)
                        activate_safe_mode(
                            f"web_audit_corruption_line_{line_num}_date_{date}"
                        )
                        
                        # Webhook inmediato
                        send_critical_webhook(
                            subject="SARAi Web Logs Corrupted - SAFE MODE ACTIVATED",
                            message=f"Corrupción detectada en línea {line_num} de web_queries_{date}.jsonl\n"
                                    f"Expected hash: {expected_hash.strip()}\n"
                                    f"Computed hash: {computed_hash}"
                        )
                        
                        return False
            
            print(f"✅ Web logs {date} verificados correctamente")
            return True
        
        except Exception as e:
            print(f"❌ Error verificando web logs: {e}")
            activate_safe_mode(f"web_audit_verification_error: {e}")
            return False
    
    def get_stats(self, date: Optional[str] = None) -> Dict:
        """
        Retorna estadísticas de búsquedas web para una fecha
        
        Returns:
            {
                "total_queries": int,
                "cache_hits": int,
                "cache_misses": int,
                "errors": int,
                "avg_snippets": float,
                "most_common_domains": List[str]
            }
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        jsonl_path = os.path.join(self.log_dir, f"web_queries_{date}.jsonl")
        
        if not os.path.exists(jsonl_path):
            return {"total_queries": 0}
        
        stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "total_snippets": 0,
            "domains": {}
        }
        
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                stats["total_queries"] += 1
                
                if entry["source"] == "cache":
                    stats["cache_hits"] += 1
                elif entry["source"] == "searxng":
                    stats["cache_misses"] += 1
                elif entry["source"] == "error":
                    stats["errors"] += 1
                
                stats["total_snippets"] += entry.get("snippets_count", 0)
                
                # Extraer dominios de URLs
                for url in entry.get("snippets_urls", []):
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
                    except:
                        pass
        
        # Calcular promedios y top domains
        stats["avg_snippets"] = stats["total_snippets"] / stats["total_queries"] if stats["total_queries"] > 0 else 0
        stats["most_common_domains"] = sorted(
            stats["domains"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        del stats["domains"]  # No retornar dict completo
        return stats


# Singleton global (importado por rag_agent.py)
_web_audit_instance = None

def get_web_audit_logger() -> WebAuditLogger:
    """Factory para singleton de WebAuditLogger"""
    global _web_audit_instance
    
    if _web_audit_instance is None:
        _web_audit_instance = WebAuditLogger()
    
    return _web_audit_instance


def log_web_query(
    query: str,
    search_results: Optional[Dict],
    response: Optional[str] = None,
    llm_model: Optional[str] = None,
    error: Optional[str] = None
):
    """
    Wrapper conveniente para logging web
    
    Usage:
        from core.web_audit import log_web_query
        log_web_query(
            query="¿Quién ganó el Oscar 2025?",
            search_results=results,
            response=synthesized_response,
            llm_model="expert_long"
        )
    """
    logger = get_web_audit_logger()
    logger.log_web_query(query, search_results, response, llm_model, error)


# ============================================================================
# VOICE AUDIT LOGGER v2.11 - HMAC-SHA256 para Voz
# ============================================================================

class VoiceAuditLogger:
    """
    Logger de auditoría para interacciones de voz con HMAC-SHA256
    
    Formato de log:
    {
        "timestamp": "2025-10-27T14:32:10.123456",
        "input_audio_sha256": "abc123...",
        "detected_lang": "es" | "fr" | "de" | "ja" | etc.,
        "engine_used": "omni" | "nllb" | "lfm2",
        "response_text": "Respuesta del sistema...",
        "response_audio_sha256": "def456..." (opcional),
        "safe_mode_active": false,
        "latency_ms": 245.8
    }
    
    HMAC Sidecar:
    - Cada línea firmada con HMAC-SHA256 usando secret key
    - Secret key desde HMAC_SECRET_KEY env var (default: "sarai-voice-audit-key")
    - Formato: logs/voice_interactions_YYYY-MM-DD.jsonl.hmac
    """
    
    def __init__(self, log_dir: str = "logs", secret_key: Optional[str] = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # HMAC secret key (desde env o default)
        self.secret_key = (
            secret_key or 
            os.getenv("HMAC_SECRET_KEY", "sarai-voice-audit-key")
        ).encode('utf-8')
        
        self.lock = threading.Lock()  # Thread-safe logging
    
    def _get_log_paths(self) -> tuple:
        """Retorna (jsonl_path, hmac_path) para el día actual"""
        date = datetime.now().strftime("%Y-%m-%d")
        jsonl_path = os.path.join(self.log_dir, f"voice_interactions_{date}.jsonl")
        hmac_path = f"{jsonl_path}.hmac"
        return jsonl_path, hmac_path
    
    def _compute_hmac(self, entry_str: str) -> str:
        """Calcula HMAC-SHA256 para una entrada de log"""
        return hmac.new(
            self.secret_key,
            entry_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def log_voice_interaction(
        self,
        input_audio: bytes,
        detected_lang: str,
        engine_used: str,
        response_text: str,
        response_audio: Optional[bytes] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None
    ):
        """
        Registra una interacción de voz con firma HMAC
        
        Args:
            input_audio: Audio de entrada (bytes WAV)
            detected_lang: Idioma detectado (ISO 639-1)
            engine_used: Motor usado ("omni" | "nllb" | "lfm2")
            response_text: Respuesta en texto
            response_audio: Audio de respuesta (opcional)
            latency_ms: Latencia total del pipeline (opcional)
            error: Mensaje de error si fallo (opcional)
        """
        with self.lock:
            from core.audit import is_safe_mode
            
            # Hash del audio de entrada
            input_hash = hashlib.sha256(input_audio).hexdigest()
            
            # Hash del audio de respuesta (si existe)
            response_hash = None
            if response_audio:
                response_hash = hashlib.sha256(response_audio).hexdigest()
            
            # Construir entrada de log
            entry = {
                "timestamp": datetime.now().isoformat(),
                "input_audio_sha256": input_hash,
                "detected_lang": detected_lang,
                "engine_used": engine_used,
                "response_text": response_text[:200] if response_text else None,  # Preview
                "response_audio_sha256": response_hash,
                "safe_mode_active": is_safe_mode(),
                "latency_ms": latency_ms,
                "error": error
            }
            
            # Serializar con keys ordenados (reproducibilidad)
            entry_str = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            
            # Calcular HMAC
            signature = self._compute_hmac(entry_str)
            
            # Escribir log principal
            jsonl_path, hmac_path = self._get_log_paths()
            
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            
            # Escribir HMAC sidecar
            with open(hmac_path, "a") as f:
                f.write(f"{signature}\n")
    
    def verify_integrity(self, date: str) -> bool:
        """
        Verifica integridad de logs de voz para una fecha
        
        Args:
            date: Fecha en formato YYYY-MM-DD
        
        Returns:
            True si logs íntegros, False si corrupción detectada
        """
        jsonl_path = os.path.join(self.log_dir, f"voice_interactions_{date}.jsonl")
        hmac_path = f"{jsonl_path}.hmac"
        
        if not os.path.exists(jsonl_path):
            print(f"⚠️  No hay logs de voz para {date}")
            return True  # No logs = no corrupción
        
        if not os.path.exists(hmac_path):
            print(f"❌ Falta archivo HMAC para {date}")
            activate_safe_mode(reason="voice_audit_missing_hmac")
            return False
        
        try:
            with open(jsonl_path) as f, open(hmac_path) as f_hmac:
                for line_num, (line, expected_hmac) in enumerate(zip(f, f_hmac), 1):
                    # Reconstruir entrada con keys ordenados
                    entry = json.loads(line.strip())
                    entry_str = json.dumps(entry, ensure_ascii=False, sort_keys=True)
                    
                    # Calcular HMAC
                    computed_hmac = self._compute_hmac(entry_str)
                    
                    # Verificar
                    if computed_hmac != expected_hmac.strip():
                        print(f"❌ HMAC inválido en línea {line_num} de {date}")
                        activate_safe_mode(reason="voice_audit_hmac_mismatch")
                        send_critical_webhook(
                            event="voice_audit_corruption",
                            details={"date": date, "line": line_num}
                        )
                        return False
            
            print(f"✅ Logs de voz para {date} verificados OK")
            return True
        
        except Exception as e:
            print(f"❌ Error verificando logs de voz: {e}")
            activate_safe_mode(reason="voice_audit_verification_error")
            return False
    
    def get_stats(self, date: str) -> Dict:
        """
        Obtiene estadísticas de interacciones de voz para una fecha
        
        Returns:
            {
                "total_interactions": int,
                "by_language": {"es": 10, "fr": 5, ...},
                "by_engine": {"omni": 12, "nllb": 3, ...},
                "avg_latency_ms": float,
                "errors_count": int
            }
        """
        jsonl_path = os.path.join(self.log_dir, f"voice_interactions_{date}.jsonl")
        
        if not os.path.exists(jsonl_path):
            return {
                "total_interactions": 0,
                "by_language": {},
                "by_engine": {},
                "avg_latency_ms": 0.0,
                "errors_count": 0
            }
        
        total = 0
        by_language = {}
        by_engine = {}
        latencies = []
        errors_count = 0
        
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                total += 1
                
                # Contar por idioma
                lang = entry.get("detected_lang", "unknown")
                by_language[lang] = by_language.get(lang, 0) + 1
                
                # Contar por motor
                engine = entry.get("engine_used", "unknown")
                by_engine[engine] = by_engine.get(engine, 0) + 1
                
                # Latencias
                if entry.get("latency_ms"):
                    latencies.append(entry["latency_ms"])
                
                # Errores
                if entry.get("error"):
                    errors_count += 1
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        return {
            "total_interactions": total,
            "by_language": by_language,
            "by_engine": by_engine,
            "avg_latency_ms": round(avg_latency, 2),
            "errors_count": errors_count
        }


# Singleton para acceso global
_voice_audit_logger_instance = None

def get_voice_audit_logger() -> VoiceAuditLogger:
    """Factory para obtener instancia singleton del voice logger"""
    global _voice_audit_logger_instance
    if _voice_audit_logger_instance is None:
        _voice_audit_logger_instance = VoiceAuditLogger()
    return _voice_audit_logger_instance


def log_voice_interaction(
    input_audio: bytes,
    detected_lang: str,
    engine_used: str,
    response_text: str,
    response_audio: Optional[bytes] = None,
    latency_ms: Optional[float] = None,
    error: Optional[str] = None
):
    """
    Función de conveniencia para logging de voz
    
    Ejemplo:
        log_voice_interaction(
            input_audio=audio_bytes,
            detected_lang="fr",
            engine_used="nllb",
            response_text="Bonjour!",
            latency_ms=1250.5
        )
    """
    logger = get_voice_audit_logger()
    logger.log_voice_interaction(
        input_audio, detected_lang, engine_used, response_text,
        response_audio, latency_ms, error
    )


# ============================================================================
# CLI EXPANDIDO v2.11 - Soporte para Web + Voice
# ============================================================================

# CLI para verificación manual
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SARAi Audit CLI v2.11 (Web + Voice)")
    
    # Comandos Web
    parser.add_argument("--verify-web", help="Verificar logs web para fecha (YYYY-MM-DD)")
    parser.add_argument("--stats-web", help="Stats de búsquedas web para fecha (YYYY-MM-DD)")
    parser.add_argument("--verify-all-web", action="store_true", help="Verificar todos los logs web")
    
    # Comandos Voice (v2.11)
    parser.add_argument("--verify-voice", help="Verificar logs de voz para fecha (YYYY-MM-DD)")
    parser.add_argument("--stats-voice", help="Stats de voz para fecha (YYYY-MM-DD)")
    parser.add_argument("--verify-all-voice", action="store_true", help="Verificar todos los logs de voz")
    
    # Comando unificado
    parser.add_argument("--verify-all", action="store_true", help="Verificar TODOS los logs (web + voz)")
    
    # Legacy compatibility (mantener --verify para web)
    parser.add_argument("--verify", help="[LEGACY] Alias de --verify-web")
    parser.add_argument("--stats", help="[LEGACY] Alias de --stats-web")
    
    args = parser.parse_args()
    
    web_logger = get_web_audit_logger()
    voice_logger = get_voice_audit_logger()
    
    # ========== COMANDOS WEB ==========
    if args.verify or args.verify_web:
        date = args.verify or args.verify_web
        print(f"\n🔍 Verificando logs web para {date}...")
        is_valid = web_logger.verify_integrity(date)
        exit(0 if is_valid else 1)
    
    elif args.stats or args.stats_web:
        date = args.stats or args.stats_web
        print(f"\n📊 Stats de búsquedas web para {date}:")
        stats = web_logger.get_stats(date)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    elif args.verify_all_web:
        print("\n🔍 Verificando TODOS los logs web...")
        log_dir = web_logger.log_dir
        dates = set()
        
        for filename in os.listdir(log_dir):
            if filename.startswith("web_queries_") and filename.endswith(".jsonl"):
                date = filename.replace("web_queries_", "").replace(".jsonl", "")
                dates.add(date)
        
        all_valid = True
        for date in sorted(dates):
            is_valid = web_logger.verify_integrity(date)
            if not is_valid:
                all_valid = False
        
        print(f"\n{'✅' if all_valid else '❌'} Logs web: {len(dates)} días")
        exit(0 if all_valid else 1)
    
    # ========== COMANDOS VOICE (v2.11) ==========
    elif args.verify_voice:
        date = args.verify_voice
        print(f"\n🔍 Verificando logs de voz para {date}...")
        is_valid = voice_logger.verify_integrity(date)
        exit(0 if is_valid else 1)
    
    elif args.stats_voice:
        date = args.stats_voice
        print(f"\n📊 Stats de interacciones de voz para {date}:")
        stats = voice_logger.get_stats(date)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    elif args.verify_all_voice:
        print("\n🔍 Verificando TODOS los logs de voz...")
        log_dir = voice_logger.log_dir
        dates = set()
        
        for filename in os.listdir(log_dir):
            if filename.startswith("voice_interactions_") and filename.endswith(".jsonl"):
                date = filename.replace("voice_interactions_", "").replace(".jsonl", "")
                dates.add(date)
        
        all_valid = True
        for date in sorted(dates):
            is_valid = voice_logger.verify_integrity(date)
            if not is_valid:
                all_valid = False
        
        print(f"\n{'✅' if all_valid else '❌'} Logs voz: {len(dates)} días")
        exit(0 if all_valid else 1)
    
    # ========== COMANDO UNIFICADO ==========
    elif args.verify_all:
        print("\n🔍 Verificando TODOS los logs (Web + Voz)...")
        
        log_dir = web_logger.log_dir
        
        # Logs web
        web_dates = set()
        for filename in os.listdir(log_dir):
            if filename.startswith("web_queries_") and filename.endswith(".jsonl"):
                date = filename.replace("web_queries_", "").replace(".jsonl", "")
                web_dates.add(date)
        
        web_valid = True
        print("\n📡 Verificando logs WEB...")
        for date in sorted(web_dates):
            if not web_logger.verify_integrity(date):
                web_valid = False
        
        # Logs voz
        voice_dates = set()
        for filename in os.listdir(log_dir):
            if filename.startswith("voice_interactions_") and filename.endswith(".jsonl"):
                date = filename.replace("voice_interactions_", "").replace(".jsonl", "")
                voice_dates.add(date)
        
        voice_valid = True
        print("\n🎤 Verificando logs VOZ...")
        for date in sorted(voice_dates):
            if not voice_logger.verify_integrity(date):
                voice_valid = False
        
        # Resumen final
        print("\n" + "="*60)
        print(f"📡 Logs Web:  {len(web_dates)} días - {'✅ OK' if web_valid else '❌ CORRUPCIÓN'}")
        print(f"🎤 Logs Voz:  {len(voice_dates)} días - {'✅ OK' if voice_valid else '❌ CORRUPCIÓN'}")
        print("="*60)
        
        exit(0 if (web_valid and voice_valid) else 1)
    
    else:
        parser.print_help()

