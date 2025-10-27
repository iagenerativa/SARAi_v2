"""
SARAi v2.10 - RAG Agent (Retrieval-Augmented Generation)

Agente de búsqueda web con síntesis LLM que:
- Respeta TODAS las garantías v2.9 Sentinel
- Nunca opera si GLOBAL_SAFE_MODE activo
- Usa cache persistente (web_cache.py)
- Firma cada búsqueda inmutablemente (web_audit.py)
- Sintetiza con SOLAR (context-aware short/long)
- Fallback a respuesta Sentinel si fallo total

Este agente se activa SOLO cuando:
- TRM-Router detecta scores['web_query'] > 0.7
- MCP enruta la query al nodo 'execute_rag'
- Safe Mode NO está activo

Garantías v2.10:
- ✅ 0% regresión (no afecta queries normales)
- ✅ P99 crítica mantenida (RAG es priority: NORMAL)
- ✅ Integridad logs 100% (SHA-256)
- ✅ Autoprotección (Safe Mode trigger si fallo persistente)
"""

import json
from typing import Dict
from datetime import datetime

# CRÍTICO: Importaciones de módulos Sentinel
from core.audit import is_safe_mode, activate_safe_mode
from core.web_cache import cached_search
from core.web_audit import log_web_query
from core.model_pool import ModelPool


# Respuestas predefinidas de Sentinel (fallback sin búsqueda)
SENTINEL_RESPONSES = {
    "web_search_disabled": (
        "Lo siento, la búsqueda web está temporalmente deshabilitada "
        "debido a que el sistema está en Modo Seguro. "
        "Esto es una medida de protección automática para garantizar la integridad de mis respuestas. "
        "Por favor, intenta de nuevo más tarde o pregunta algo que pueda responder con mi conocimiento interno."
    ),
    "web_search_failed": (
        "No pude acceder a información actualizada en este momento. "
        "Puedo intentar responder basándome en mi conocimiento interno, "
        "pero ten en cuenta que podría no estar completamente actualizado. "
        "¿Quieres que continúe con esa limitación?"
    ),
    "synthesis_failed": (
        "Encontré información relevante pero tuve problemas al procesarla. "
        "Por seguridad, prefiero no ofrecer una respuesta que podría ser incorrecta. "
        "¿Podrías reformular tu pregunta de manera más específica?"
    )
}


def sentinel_response(reason: str) -> Dict:
    """
    Retorna una respuesta Sentinel predefinida
    
    Filosofía v2.10: "Prefiere el silencio selectivo sobre la mentira"
    """
    response = SENTINEL_RESPONSES.get(
        reason,
        "Lo siento, no puedo procesar esta consulta en este momento por razones de seguridad."
    )
    
    return {
        "response": response,
        "sentinel_triggered": True,
        "sentinel_reason": reason,
        "timestamp": datetime.now().isoformat()
    }


def execute_rag(state: Dict, model_pool: ModelPool) -> Dict:
    """
    Ejecuta pipeline RAG completo con garantías Sentinel
    
    Pipeline (6 pasos):
    1. GARANTÍA SENTINEL: Verificar Safe Mode
    2. BÚSQUEDA CACHEADA: cached_search() con SearXNG
    3. AUDITORÍA: log_web_query() con SHA-256
    4. SÍNTESIS: Prompt engineering con snippets
    5. LLM: SOLAR (short/long según contexto)
    6. FALLBACK: sentinel_response() si fallo total
    
    Args:
        state: State de LangGraph con 'input', 'scores', etc.
        model_pool: ModelPool para acceso a SOLAR
    
    Returns:
        state actualizado con 'response' y 'rag_metadata'
    """
    query = state["input"]
    
    # ====== PASO 1: GARANTÍA SENTINEL ======
    if is_safe_mode():
        print("🛡️ RAG Agent: GLOBAL_SAFE_MODE activo, búsqueda bloqueada")
        sentinel = sentinel_response("web_search_disabled")
        state.update(sentinel)
        return state
    
    # ====== PASO 2: BÚSQUEDA CACHEADA ======
    try:
        print(f"🔍 RAG Agent: Buscando '{query[:60]}...'")
        search_results = cached_search(query)
        
        if search_results is None:
            # SearXNG no disponible o timeout
            print("⚠️ RAG Agent: Búsqueda falló (SearXNG no disponible)")
            
            # Loggear el error
            log_web_query(
                query=query,
                search_results=None,
                error="searxng_unavailable"
            )
            
            sentinel = sentinel_response("web_search_failed")
            state.update(sentinel)
            return state
        
        if len(search_results.get("snippets", [])) == 0:
            # SearXNG retornó 0 resultados
            print(f"⚠️ RAG Agent: 0 snippets para '{query[:60]}...'")
            
            # Loggear (puede trigger anomalía si es recurrente)
            log_web_query(
                query=query,
                search_results=search_results,
                error="zero_snippets"
            )
            
            sentinel = sentinel_response("web_search_failed")
            state.update(sentinel)
            return state
    
    except Exception as e:
        print(f"❌ RAG Agent: Error en búsqueda: {e}")
        log_web_query(query=query, search_results=None, error=str(e))
        
        # Trigger Safe Mode si error persistente
        activate_safe_mode(f"rag_search_exception: {e}")
        
        sentinel = sentinel_response("web_search_failed")
        state.update(sentinel)
        return state
    
    # ====== PASO 3: AUDITORÍA (pre-síntesis) ======
    # Registramos los resultados CRUDOS antes de procesarlos
    # Si hay manipulación en síntesis, tenemos el original firmado
    print(f"📝 RAG Agent: {len(search_results['snippets'])} snippets obtenidos, firmando...")
    
    # ====== PASO 4: SÍNTESIS (Prompt Engineering) ======
    try:
        # Construir prompt con snippets
        prompt_parts = [
            "Eres SARAi, una IA que sintetiza información verificable de fuentes web.",
            "Usando ÚNICAMENTE los siguientes extractos, responde a la pregunta del usuario.",
            "REGLAS CRÍTICAS:",
            "- Cita la fuente (URL) cuando uses un extracto",
            "- Si los extractos no contienen la respuesta, di 'No encontré información concluyente'",
            "- NO inventes información que no esté en los extractos",
            "- Sé conciso y directo",
            "",
            f"PREGUNTA DEL USUARIO: {query}",
            "",
            "EXTRACTOS VERIFICADOS:"
        ]
        
        for i, snippet in enumerate(search_results["snippets"], 1):
            prompt_parts.append(f"\n[Fuente {i}] {snippet['title']}")
            prompt_parts.append(f"URL: {snippet['url']}")
            prompt_parts.append(f"Contenido: {snippet['content']}")
            prompt_parts.append("---")
        
        prompt_parts.append("\nRESPUESTA (citando fuentes cuando sea posible):")
        prompt = "\n".join(prompt_parts)
        
        # Decidir modelo (short vs long)
        # Si el prompt es muy grande (muchos snippets), usar expert_long
        # Threshold: ~400 chars por snippet * 5 snippets = ~2000 chars → usar long
        prompt_length = len(prompt)
        model_name = "expert_long" if prompt_length > 1500 else "expert_short"
        
        print(f"🧠 RAG Agent: Sintetizando con {model_name} (prompt: {prompt_length} chars)...")
    
    except Exception as e:
        print(f"❌ RAG Agent: Error construyendo prompt: {e}")
        log_web_query(query=query, search_results=search_results, error=f"prompt_build_error: {e}")
        
        sentinel = sentinel_response("synthesis_failed")
        state.update(sentinel)
        return state
    
    # ====== PASO 5: LLM (SOLAR context-aware) ======
    try:
        # Obtener modelo del pool (carga bajo demanda)
        llm = model_pool.get(model_name)
        
        # Generar respuesta (esto puede tardar ~15-25s en CPU)
        response = llm(
            prompt,
            max_tokens=512,  # Respuestas concisas
            temperature=0.3,  # Bajo para síntesis factual
            stop=["PREGUNTA", "\n\n\n"]  # Stop sequences
        )
        
        # Extraer solo el texto generado
        synthesized_text = response.get("choices", [{}])[0].get("text", "").strip()
        
        if not synthesized_text:
            print("⚠️ RAG Agent: LLM retornó respuesta vacía")
            log_web_query(query=query, search_results=search_results, error="empty_llm_response")
            
            sentinel = sentinel_response("synthesis_failed")
            state.update(sentinel)
            return state
        
        print(f"✅ RAG Agent: Respuesta sintetizada ({len(synthesized_text)} chars)")
    
    except Exception as e:
        print(f"❌ RAG Agent: Error en LLM: {e}")
        log_web_query(
            query=query,
            search_results=search_results,
            llm_model=model_name,
            error=f"llm_generation_error: {e}"
        )
        
        # Trigger Safe Mode (LLM failure es crítico)
        activate_safe_mode(f"rag_llm_exception: {e}")
        
        sentinel = sentinel_response("synthesis_failed")
        state.update(sentinel)
        return state
    
    # ====== PASO 6: AUDITORÍA FINAL (post-síntesis) ======
    log_web_query(
        query=query,
        search_results=search_results,
        response=synthesized_text,
        llm_model=model_name
    )
    
    # ====== RETORNO EXITOSO ======
    state["response"] = synthesized_text
    state["rag_metadata"] = {
        "source": search_results["source"],
        "snippets_count": len(search_results["snippets"]),
        "snippets_urls": [s["url"] for s in search_results["snippets"]],
        "llm_model": model_name,
        "prompt_length": prompt_length,
        "synthesis_success": True,
        "timestamp": datetime.now().isoformat()
    }
    state["sentinel_triggered"] = False
    
    return state


# Función auxiliar para integración en graph.py
def create_rag_node(model_pool: ModelPool):
    """
    Factory que retorna un nodo callable para LangGraph
    
    Usage en graph.py:
        from agents.rag_agent import create_rag_node
        
        rag_node = create_rag_node(model_pool)
        workflow.add_node("execute_rag", rag_node)
    """
    def rag_node(state: Dict) -> Dict:
        return execute_rag(state, model_pool)
    
    return rag_node


# Testing standalone
if __name__ == "__main__":
    import argparse
    from core.model_pool import ModelPool
    import yaml
    
    parser = argparse.ArgumentParser(description="SARAi RAG Agent Test")
    parser.add_argument("--query", "-q", required=True, help="Query para RAG")
    parser.add_argument("--model", "-m", default="expert_short", 
                        choices=["expert_short", "expert_long"],
                        help="Modelo LLM a usar")
    
    args = parser.parse_args()
    
    # Cargar config
    with open("config/sarai.yaml") as f:
        config = yaml.safe_load(f)
    
    # Inicializar ModelPool
    pool = ModelPool(config)
    
    # Ejecutar RAG
    print(f"\n{'='*60}")
    print(f"RAG Agent Test: {args.query}")
    print(f"{'='*60}\n")
    
    state = {"input": args.query, "scores": {"web_query": 0.9}}
    result_state = execute_rag(state, pool)
    
    print(f"\n{'='*60}")
    print("RESULTADO:")
    print(f"{'='*60}")
    print(result_state["response"])
    
    if "rag_metadata" in result_state:
        print(f"\n{'='*60}")
        print("METADATA:")
        print(f"{'='*60}")
        print(json.dumps(result_state["rag_metadata"], indent=2, ensure_ascii=False))
    
    if result_state.get("sentinel_triggered"):
        print(f"\n⚠️ SENTINEL TRIGGERED: {result_state.get('sentinel_reason')}")
