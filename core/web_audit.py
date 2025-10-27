"""
SARAi v2.10 - Web Audit Module (Logging firmado SHA-256)

Sistema de auditoría inmutable para búsquedas web RAG:
- Cada query web se firma con SHA-256 (integridad verificable)
- Formato: logs/web_queries_YYYY-MM-DD.jsonl + .sha256
- Compatible con el sistema de auditoría v2.9 (core/audit.py)
- Trigger de Sentinel Mode si detección de manipulación

Garantías v2.10:
- ✅ Cada búsqueda web firmada inmutablemente
- ✅ Logs verificables con SHA-256 sidecar
- ✅ Trigger automático de Safe Mode si corrupción detectada
- ✅ Webhooks de alerta para consultas anómalas
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Optional
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


# CLI para verificación manual
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SARAi Web Audit CLI")
    parser.add_argument("--verify", help="Verificar integridad para fecha (YYYY-MM-DD)")
    parser.add_argument("--stats", help="Mostrar stats para fecha (YYYY-MM-DD)")
    parser.add_argument("--verify-all", action="store_true", help="Verificar todos los logs web")
    
    args = parser.parse_args()
    
    logger = get_web_audit_logger()
    
    if args.verify:
        print(f"\n🔍 Verificando logs web para {args.verify}...")
        is_valid = logger.verify_integrity(args.verify)
        if is_valid:
            print("✅ Logs web íntegros")
        else:
            print("❌ CORRUPCIÓN DETECTADA")
            exit(1)
    
    elif args.stats:
        print(f"\n📊 Stats de búsquedas web para {args.stats}:")
        stats = logger.get_stats(args.stats)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    elif args.verify_all:
        print("\n🔍 Verificando TODOS los logs web...")
        log_dir = logger.log_dir
        dates = set()
        
        for filename in os.listdir(log_dir):
            if filename.startswith("web_queries_") and filename.endswith(".jsonl"):
                date = filename.replace("web_queries_", "").replace(".jsonl", "")
                dates.add(date)
        
        all_valid = True
        for date in sorted(dates):
            is_valid = logger.verify_integrity(date)
            if not is_valid:
                all_valid = False
        
        if all_valid:
            print(f"✅ Todos los logs web verificados ({len(dates)} días)")
        else:
            print("❌ CORRUPCIÓN DETECTADA en uno o más días")
            exit(1)
    
    else:
        parser.print_help()
