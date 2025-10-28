# Arquitectura LangGraph de SARAi v2.11

**Respuesta rápida**: **SÍ, SARAi está 100% optimizado con LangGraph** desde el inicio del proyecto.

---

## 🏗️ Arquitectura Completa

SARAi usa **LangGraph** (del ecosistema LangChain) como orquestador central de todo el flujo de procesamiento. No es un "añadido posterior", es el **núcleo arquitectónico** del sistema.

### Stack LangChain Completo

```yaml
# requirements.txt
langchain>=0.1.0          # Framework base
langchain-core>=0.1.0     # Abstracciones core (Runnable, etc.)
langgraph>=0.0.40         # Orquestación de estado
```

---

## 📊 Grafo de Estado (StateGraph)

El flujo completo de SARAi está definido como un **StateGraph** en `core/graph.py`:

```python
from langgraph.graph import StateGraph, END

class State(TypedDict):
    """Estado compartido en el flujo LangGraph"""
    input: str                    # Query del usuario
    hard: float                   # Score técnico (TRM)
    soft: float                   # Score emocional (TRM)
    web_query: float              # Score búsqueda web (v2.10)
    alpha: float                  # Peso hard (MCP)
    beta: float                   # Peso soft (MCP)
    agent_used: str               # "expert" | "tiny" | "rag"
    response: str                 # Respuesta final
    feedback: float               # Feedback implícito
    rag_metadata: dict            # Metadata RAG (v2.10)
```

### Flujo de Nodos (v2.10)

```
┌─────────────────────────────────────────────────────────────┐
│                       ENTRADA (Input)                        │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
                   ┌───────────────┐
                   │   classify    │  TRM-Router + EmbeddingGemma
                   │ (hard/soft/   │  Clasifica intención
                   │  web_query)   │
                   └───────┬───────┘
                           ↓
                   ┌───────────────┐
                   │      mcp      │  Meta Control Plane
                   │  (α, β pesos) │  Calcula pesos dinámicos
                   └───────┬───────┘
                           ↓
                  ┌────────────────┐
                  │ ROUTING LOGIC  │
                  │ (condicional)  │
                  └────┬───┬───┬───┘
                       ↓   ↓   ↓
        ┌──────────────┘   │   └──────────────┐
        ↓                  ↓                  ↓
┌───────────────┐  ┌───────────────┐  ┌──────────────┐
│generate_expert│  │ execute_rag   │  │generate_tiny │
│  (SOLAR LLM)  │  │(Búsqueda web) │  │  (LFM2 LLM)  │
└───────┬───────┘  └───────┬───────┘  └──────┬───────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                           ↓
                   ┌───────────────┐
                   │   feedback    │  Logger asíncrono
                   │  (implícito)  │  Aprendizaje continuo
                   └───────┬───────┘
                           ↓
                          END
```

---

## 🔀 Routing Condicional (add_conditional_edges)

El poder de LangGraph está en el **routing dinámico** basado en el estado:

```python
# core/graph.py
workflow.add_conditional_edges(
    "mcp",
    self._route_to_agent,
    {
        "expert": "generate_expert",
        "tiny": "generate_tiny",
        "rag": "execute_rag"
    }
)

def _route_to_agent(self, state: State) -> str:
    """
    Lógica de decisión inteligente:
    1. Si web_query > 0.7 → RAG (búsqueda web)
    2. Si alpha > 0.7 → Expert (SOLAR)
    3. Else → Tiny (LFM2)
    """
    if state.get("web_query", 0.0) > 0.7:
        return "rag"
    
    if state["alpha"] > 0.7:
        return "expert"
    
    return "tiny"
```

**Ventajas sobre flujos lineales**:
- ✅ **Decisiones dinámicas**: No todos los queries pasan por todos los LLMs
- ✅ **Eficiencia RAM**: Solo carga el modelo necesario
- ✅ **Latencia optimizada**: Rutas cortas (tiny) vs largas (expert)
- ✅ **Extensibilidad**: Añadir nodos nuevos sin romper flujo

---

## 🧩 Integración de Componentes

Cada componente de SARAi es un **nodo LangGraph**:

### 1. Nodo: classify (TRM-Router)

```python
def _classify_intent(self, state: State) -> dict:
    """Nodo: Clasificar hard/soft/web_query intent"""
    user_input = state["input"]
    
    # TRM-Router usa embeddings para clasificar
    embedding = self.embedding_model.encode(user_input)
    scores = self.trm_classifier.invoke(embedding_tensor)
    
    return {
        "hard": scores["hard"],
        "soft": scores["soft"],
        "web_query": scores.get("web_query", 0.0)
    }
```

**Output**: Actualiza `state["hard"]`, `state["soft"]`, `state["web_query"]`

### 2. Nodo: mcp (Meta Control Plane)

```python
def _compute_weights(self, state: State) -> dict:
    """Nodo: Calcular pesos α/β con MCP"""
    alpha, beta = self.mcp.compute_weights(
        state["hard"], 
        state["soft"]
    )
    
    return {"alpha": alpha, "beta": beta}
```

**Output**: Actualiza `state["alpha"]`, `state["beta"]`

### 3. Nodo: generate_expert (SOLAR)

```python
def _generate_expert(self, state: State) -> dict:
    """Nodo: Generar respuesta técnica con SOLAR"""
    from core.model_pool import get_model_pool
    
    pool = get_model_pool()
    solar = pool.get("expert_short")  # o "expert_long"
    
    response = solar(state["input"], max_tokens=512)
    
    return {
        "response": response["choices"][0]["text"],
        "agent_used": "expert"
    }
```

**Output**: Actualiza `state["response"]`, `state["agent_used"]`

### 4. Nodo: execute_rag (Búsqueda Web + LLM)

```python
# agents/rag_agent.py
def create_rag_node(model_pool):
    """Factory para nodo RAG compatible con LangGraph"""
    
    def rag_node(state: State) -> dict:
        # 1. Búsqueda web
        results = cached_search(state["input"])
        
        # 2. Síntesis con LLM
        prompt = build_rag_prompt(state["input"], results)
        llm = model_pool.get("expert_long")
        response = llm(prompt)
        
        # 3. Auditoría
        log_web_query(state["input"], results, response)
        
        return {
            "response": response,
            "agent_used": "rag",
            "rag_metadata": {
                "snippets_count": len(results),
                "source": "searxng"
            }
        }
    
    return rag_node
```

**Output**: Actualiza `state["response"]`, `state["rag_metadata"]`

---

## 🎯 Ventajas de LangGraph en SARAi

| Aspecto | Sin LangGraph | Con LangGraph (SARAi) |
|---------|---------------|------------------------|
| **Control de flujo** | if/else anidados, difícil seguir | Grafo visual, fácil debuggear |
| **Estado compartido** | Pasar dictionaries manualmente | TypedDict automático |
| **Routing** | Lógica dispersa en funciones | `add_conditional_edges` declarativo |
| **Debugging** | Print statements, difícil trace | LangSmith, visualización de grafo |
| **Testing** | Mockear cada función individualmente | Testear nodos independientes |
| **Extensibilidad** | Modificar flujo = refactor masivo | Añadir nodo = 3 líneas |
| **Persistencia** | Implementar manualmente | Checkpointing built-in |
| **Paralelización** | Threading manual complejo | `add_parallel_edges` automático |

---

## 🔄 Flujo Completo (Ejemplo Real)

**Input**: "¿Cómo está el clima en Tokio?"

```
1. classify:
   - hard: 0.2
   - soft: 0.1
   - web_query: 0.9  ← DETECTA búsqueda web

2. mcp:
   - alpha: 0.3
   - beta: 0.7
   (No importa porque web_query > 0.7)

3. _route_to_agent:
   - Decisión: "rag"  ← Va a búsqueda web

4. execute_rag:
   - cached_search("¿Cómo está el clima en Tokio?")
   - Encuentra 5 snippets de weather.com
   - Síntesis con SOLAR
   - response: "Según weather.com, el clima actual en Tokio es..."
   - rag_metadata: {snippets_count: 5, source: "searxng"}

5. feedback:
   - log_web_query(...) → Auditoría HMAC
   - Feedback implícito detectado (continuación de diálogo)

6. END
```

**Latencia total**: ~25-30s (P50 RAG)

---

## 🧪 Testing con LangGraph

LangGraph facilita enormemente el testing:

```python
# tests/test_graph.py
def test_web_query_routing():
    """Verifica que web_query > 0.7 enruta a RAG"""
    
    orchestrator = SARAiOrchestrator()
    
    # Estado inicial
    state = {
        "input": "¿Cómo está el clima en Madrid?",
        "hard": 0.2,
        "soft": 0.1,
        "web_query": 0.9,  # ← Forzar web_query alto
        "alpha": 0.5,
        "beta": 0.5
    }
    
    # Ejecutar solo el routing
    route = orchestrator._route_to_agent(state)
    
    assert route == "rag"  # ✅ Correcto
```

**Sin LangGraph**, tendrías que mockear todo el flujo completo.

---

## 📚 Recursos LangChain/LangGraph

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **LangChain Core**: https://python.langchain.com/docs/
- **LangSmith** (debugging): https://smith.langchain.com/
- **StateGraph Tutorial**: https://langchain-ai.github.io/langgraph/tutorials/introduction/

---

## 🎓 Patrón de Diseño: State Machine como Grafo

SARAi implementa el patrón **Finite State Machine** (FSM) usando **StateGraph**:

```
Estados:
- CLASSIFY: Analiza intención
- MCP: Calcula pesos
- EXPERT: Genera con SOLAR
- TINY: Genera con LFM2
- RAG: Busca en web + sintetiza
- FEEDBACK: Aprende

Transiciones:
- CLASSIFY → MCP (siempre)
- MCP → EXPERT (si alpha > 0.7)
- MCP → TINY (si alpha <= 0.7 y web_query <= 0.7)
- MCP → RAG (si web_query > 0.7)
- [EXPERT|TINY|RAG] → FEEDBACK (siempre)
- FEEDBACK → END (siempre)
```

**Ventaja clave**: Cada estado (nodo) es **stateless** individualmente, pero el grafo mantiene el estado global.

---

## 🚀 Conclusión

**SARAi NO es un proyecto "con LangGraph añadido"**.

**SARAi ES un proyecto LangGraph desde su concepción**.

Todo el sistema (TRM, MCP, LLMs, RAG, feedback) está orquestado mediante:
- ✅ **StateGraph**: Flujo de estado
- ✅ **TypedDict**: Estado tipado
- ✅ **add_conditional_edges**: Routing inteligente
- ✅ **Nodos independientes**: Modularidad extrema
- ✅ **Runnable Protocol**: Interop con LangChain

**Si quitas LangGraph de SARAi, el sistema deja de existir**.

---

**Última actualización**: 2025-10-28  
**Versión SARAi**: v2.11  
**LangGraph versión**: >=0.0.40
