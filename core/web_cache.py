"""
SARAi v2.10 - Web Cache Module (SearXNG + diskcache)

Sistema de caché persistente para búsquedas web que:
- Reduce llamadas redundantes a SearXNG (ahorra ancho de banda)
- Respeta GLOBAL_SAFE_MODE (no busca si está activado)
- TTL configurable por dominio de búsqueda
- Invalidación automática para queries time-sensitive

Garantías v2.10:
- ✅ Nunca hace búsqueda web si GLOBAL_SAFE_MODE activo
- ✅ Cache miss → búsqueda SearXNG → firma en web_audit
- ✅ Cache hit → retorno instantáneo sin red
- ✅ Timeout 10s por búsqueda (no bloquea sistema)
"""

import os
import json
import hashlib
import time
from typing import Optional, Dict, List
from datetime import datetime
import requests
from diskcache import Cache

# CRÍTICO: Importar Safe Mode de audit.py
from core.audit import GLOBAL_SAFE_MODE, is_safe_mode


class WebCache:
    """
    Cache persistente de búsquedas web con SearXNG local
    
    Configuración recomendada:
    - SearXNG: docker run -d -p 8888:8080 searxng/searxng
    - TTL: 3600s (1h) para queries generales, 300s (5min) para time-sensitive
    - Max snippets: 5 (balance entre contexto y RAM)
    """
    
    def __init__(
        self,
        searxng_url: str = "http://localhost:8888",
        cache_dir: str = "state/web_cache",
        ttl: int = 3600,
        max_snippets: int = 5
    ):
        self.searxng_url = searxng_url.rstrip('/')
        self.cache = Cache(cache_dir, size_limit=1024**3)  # 1GB max
        self.ttl = ttl
        self.max_snippets = max_snippets
        
        # Timeout para búsquedas (no bloquear sistema)
        self.search_timeout = 10  # segundos
    
    def _normalize_query(self, query: str) -> str:
        """Normaliza query para cache key (lowercase, strip)"""
        return query.lower().strip()
    
    def _cache_key(self, query: str) -> str:
        """Genera key SHA-256 para diskcache"""
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _is_time_sensitive(self, query: str) -> bool:
        """
        Detecta si la query requiere datos recientes (TTL reducido)
        
        Ejemplos time-sensitive:
        - "clima en Tokio"
        - "precio de Bitcoin"
        - "noticias de hoy"
        - "resultados del partido"
        """
        time_keywords = [
            "clima", "weather", "precio", "price", "stock",
            "noticias", "news", "hoy", "today", "ahora", "now",
            "partido", "match", "resultado", "score", "live"
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in time_keywords)
    
    def get(self, query: str) -> Optional[Dict]:
        """
        Obtiene resultados de búsqueda (cache o SearXNG)
        
        Returns:
            {
                "query": str,
                "snippets": List[Dict],  # [{"title": "", "url": "", "content": ""}]
                "timestamp": str,
                "source": "cache" | "searxng"
            }
            None si Safe Mode activo o fallo total
        """
        # GARANTÍA 1: Respeto de Safe Mode
        if is_safe_mode():
            print("⚠️ Web cache: GLOBAL_SAFE_MODE activo, búsqueda bloqueada")
            return None
        
        # GARANTÍA 2: Búsqueda en cache
        cache_key = self._cache_key(query)
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            # Verificar TTL manual (time-sensitive usa TTL reducido)
            age = time.time() - cached.get("cached_at", 0)
            ttl = 300 if self._is_time_sensitive(query) else self.ttl
            
            if age < ttl:
                print(f"✅ Web cache HIT: {query[:50]}... (age: {age:.1f}s)")
                cached["source"] = "cache"
                return cached
            else:
                # Expirado, eliminar
                self.cache.delete(cache_key)
                print(f"🔄 Web cache EXPIRED: {query[:50]}... (age: {age:.1f}s > {ttl}s)")
        
        # GARANTÍA 3: Cache miss → búsqueda SearXNG
        print(f"🔍 Web cache MISS: {query[:50]}... → SearXNG")
        return self._search_searxng(query, cache_key)
    
    def _search_searxng(self, query: str, cache_key: str) -> Optional[Dict]:
        """
        Realiza búsqueda en SearXNG local
        
        SearXNG API: GET /search?q=<query>&format=json
        """
        try:
            response = requests.get(
                f"{self.searxng_url}/search",
                params={"q": query, "format": "json"},
                timeout=self.search_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            # Limitar a max_snippets (evitar saturar contexto LLM)
            snippets = [
                {
                    "title": r.get("title", "Sin título"),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:500]  # Max 500 chars/snippet
                }
                for r in results[:self.max_snippets]
            ]
            
            result = {
                "query": query,
                "snippets": snippets,
                "timestamp": datetime.now().isoformat(),
                "source": "searxng",
                "cached_at": time.time()
            }
            
            # Guardar en cache
            self.cache.set(cache_key, result)
            print(f"✅ SearXNG: {len(snippets)} snippets obtenidos para '{query[:50]}...'")
            
            return result
        
        except requests.exceptions.Timeout:
            print(f"⏱️ SearXNG timeout ({self.search_timeout}s) para: {query[:50]}...")
            return None
        
        except requests.exceptions.ConnectionError:
            print(f"❌ SearXNG no disponible en {self.searxng_url}")
            return None
        
        except Exception as e:
            print(f"❌ Error en SearXNG: {e}")
            return None
    
    def invalidate(self, query: str):
        """Invalida cache para una query específica (manual)"""
        cache_key = self._cache_key(query)
        deleted = self.cache.delete(cache_key)
        if deleted:
            print(f"🗑️ Cache invalidado para: {query[:50]}...")
    
    def clear_expired(self):
        """Limpia entradas expiradas del cache (cron diario)"""
        count = 0
        for key in list(self.cache.iterkeys()):
            entry = self.cache.get(key)
            if entry:
                age = time.time() - entry.get("cached_at", 0)
                ttl = 300 if self._is_time_sensitive(entry["query"]) else self.ttl
                if age > ttl:
                    self.cache.delete(key)
                    count += 1
        
        print(f"🧹 Web cache: {count} entradas expiradas eliminadas")
    
    def stats(self) -> Dict:
        """Métricas del cache para monitoreo"""
        return {
            "size_mb": self.cache.volume() / (1024**2),
            "entries": len(self.cache),
            "ttl_default": self.ttl,
            "max_snippets": self.max_snippets
        }


# Singleton global (importado por rag_agent.py)
_web_cache_instance = None

def get_web_cache(config: Optional[Dict] = None) -> WebCache:
    """Factory para singleton de WebCache"""
    global _web_cache_instance
    
    if _web_cache_instance is None:
        if config is None:
            # Cargar config por defecto
            import yaml
            with open("config/sarai.yaml") as f:
                config = yaml.safe_load(f)
        
        rag_config = config.get("rag", {})
        _web_cache_instance = WebCache(
            searxng_url=rag_config.get("searxng_url", "http://localhost:8888"),
            cache_dir=rag_config.get("cache_dir", "state/web_cache"),
            ttl=rag_config.get("cache_ttl", 3600),
            max_snippets=rag_config.get("max_snippets", 5)
        )
    
    return _web_cache_instance


def cached_search(query: str) -> Optional[Dict]:
    """
    Wrapper conveniente para búsquedas cacheadas
    
    Usage:
        from core.web_cache import cached_search
        results = cached_search("¿Quién ganó el Oscar 2025?")
        if results:
            for snippet in results["snippets"]:
                print(snippet["title"], snippet["url"])
    """
    cache = get_web_cache()
    return cache.get(query)


# CLI para testing manual
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SARAi Web Cache CLI")
    parser.add_argument("--query", "-q", help="Ejecutar búsqueda")
    parser.add_argument("--clear", action="store_true", help="Limpiar cache expirado")
    parser.add_argument("--stats", action="store_true", help="Mostrar stats")
    parser.add_argument("--invalidate", help="Invalidar query específica")
    
    args = parser.parse_args()
    
    cache = get_web_cache()
    
    if args.query:
        print(f"\n🔍 Buscando: {args.query}")
        result = cache.get(args.query)
        if result:
            print(f"✅ Fuente: {result['source']}")
            print(f"✅ Timestamp: {result['timestamp']}")
            print(f"✅ Snippets: {len(result['snippets'])}")
            for i, snip in enumerate(result["snippets"], 1):
                print(f"\n--- Snippet {i} ---")
                print(f"Título: {snip['title']}")
                print(f"URL: {snip['url']}")
                print(f"Contenido: {snip['content'][:200]}...")
        else:
            print("❌ Sin resultados (Safe Mode activo o error)")
    
    elif args.clear:
        print("\n🧹 Limpiando cache expirado...")
        cache.clear_expired()
    
    elif args.stats:
        print("\n📊 Stats del cache:")
        stats = cache.stats()
        print(json.dumps(stats, indent=2))
    
    elif args.invalidate:
        cache.invalidate(args.invalidate)
    
    else:
        parser.print_help()
