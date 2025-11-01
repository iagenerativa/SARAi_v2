# 🎯 Plan Maestro de Consolidación: v2.12 → v2.18
**Fecha Inicio**: 31 Octubre 2025  
**Objetivo**: Implementar COMPLETO todas las versiones faltantes  
**Método**: Consolidación secuencial, validación en cada etapa

---

## 📋 Índice de Implementación

1. **FASE 1**: v2.12 Phoenix Integration (Skills en Graph) - 1-2 días
2. **FASE 2**: v2.13 Layer Architecture Integration - 2-3 días
3. **FASE 3**: v2.14 Patch Sandbox (Skills-as-Services) - 3-4 días
4. **FASE 4**: v2.15 Sentience (GPG Signer + Auditoría) - 2-3 días
5. **FASE 5**: v2.16 Omni Loop (Reflexive AGI) - 4-5 días
6. **FASE 6**: v2.17 Threading (Legacy) - SKIP (ya en v2.18)
7. **FASE 7**: v2.18 Full-Duplex - VALIDACIÓN FINAL

**Timeline Total**: ~15-20 días de trabajo enfocado

---

## 🚀 FASE 1: v2.12 Phoenix Integration (DÍA 1-2)

### Estado Actual
- ✅ Skills system implementado (7 skills, 38 tests)
- ❌ NO integrado en `core/graph.py`
- ❌ NO aplicado automáticamente en ejecución

### Objetivos
1. Integrar `detect_and_apply_skill()` en nodos del grafo
2. Skills se aplican automáticamente según keywords
3. Tests end-to-end validando aplicación de skills

### Tareas Específicas

#### Tarea 1.1: Modificar `core/graph.py` para Skill Detection
**Archivo**: `core/graph.py`  
**Cambios**:
```python
# ANTES
def generate_expert_node(state: State):
    solar = model_pool.get("expert_long" if len(state["input"]) > 400 else "expert_short")
    response = solar.generate(state["input"])
    return {"response": response}

# DESPUÉS
def generate_expert_node(state: State):
    # 1. Detectar skill aplicable
    from core.mcp import detect_and_apply_skill
    skill_config = detect_and_apply_skill(state["input"], "solar")
    
    if skill_config:
        # 2. Aplicar prompt especializado
        prompt = skill_config["full_prompt"]
        params = skill_config["generation_params"]
        
        # 3. Usar modelo recomendado si difiere
        model_name = "expert_long" if len(state["input"]) > 400 else "expert_short"
        solar = model_pool.get(model_name)
        
        # 4. Generar con parámetros optimizados
        response = solar.generate(prompt, **params)
        
        # 5. Log skill usado
        state["skill_used"] = skill_config["skill_name"]
    else:
        # Fallback: prompt estándar
        solar = model_pool.get("expert_long" if len(state["input"]) > 400 else "expert_short")
        response = solar.generate(state["input"])
        state["skill_used"] = None
    
    return {"response": response}
```

**LOC Estimado**: +40  
**Tests**: Modificar `tests/test_graph.py`  
**Tiempo**: 2-3 horas

---

#### Tarea 1.2: Modificar `generate_tiny_node` para Skills
**Archivo**: `core/graph.py`  
**Similar a Tarea 1.1 pero para LFM2**

**Nota**: Creative skill prefiere LFM2 → routing inteligente

**LOC Estimado**: +35  
**Tiempo**: 1-2 horas

---

#### Tarea 1.3: Tests End-to-End con Skills
**Archivo**: `tests/test_graph_skills_integration.py` (NUEVO)

**Tests a crear**:
```python
def test_programming_skill_auto_applied():
    """Verifica que query de código activa programming skill"""
    query = "Implementa quicksort en Python"
    state = graph.run({"input": query})
    
    assert state["skill_used"] == "programming"
    assert "def quicksort" in state["response"].lower()

def test_creative_skill_uses_lfm2():
    """Verifica que creative skill usa LFM2"""
    query = "Crea una historia corta sobre un robot"
    state = graph.run({"input": query})
    
    assert state["skill_used"] == "creative"
    assert state["model_used"] == "lfm2"

def test_no_skill_fallback():
    """Verifica fallback cuando no hay skill aplicable"""
    query = "Hola, ¿cómo estás?"
    state = graph.run({"input": query})
    
    assert state["skill_used"] is None
```

**LOC Estimado**: +200  
**Tests**: 8-10 tests  
**Tiempo**: 3-4 horas

---

#### Tarea 1.4: Logging de Skills Aplicados
**Archivo**: `core/feedback.py`

**Agregar campo skill_used al log**:
```python
entry = {
    "timestamp": datetime.now().isoformat(),
    "input": state["input"],
    "hard": state["hard"],
    "soft": state["soft"],
    "alpha": state["alpha"],
    "beta": state["beta"],
    "skill_used": state.get("skill_used", None),  # NEW
    "response": state["response"],
    "feedback": None
}
```

**LOC Estimado**: +5  
**Tiempo**: 30 minutos

---

### Entregables Fase 1
- [x] `core/graph.py` modificado con skill detection
- [x] `tests/test_graph_skills_integration.py` completo (8-10 tests)
- [x] `core/feedback.py` loggeando skills usados
- [x] Documentación actualizada en `docs/SKILLS_INTEGRATION.md`

**Criterio de Éxito**:
- ✅ Tests end-to-end pasando (8-10 tests)
- ✅ Skills se aplican automáticamente en >80% de casos relevantes
- ✅ Logs muestran skill_used correctamente

**Tiempo Total Fase 1**: **8-12 horas** (1-2 días)

---

## 🏗️ FASE 2: v2.13 Layer Architecture Integration (DÍA 3-5)

### Estado Actual
- ✅ Archivos Layer1-3 creados
- ❌ NO integrados en `core/graph.py`
- ❌ Propósito de cada layer no documentado

### Objetivos
1. Documentar arquitectura de layers
2. Integrar Layer1 I/O con graph
3. Conectar Layer2 Memory con MCP
4. Activar Layer3 Fluidity

### Tareas Específicas

#### Tarea 2.1: Documentar Arquitectura de Layers
**Archivo**: `docs/LAYER_ARCHITECTURE.md` (NUEVO)

**Contenido**:
```markdown
# Arquitectura de 3 Layers - SARAi v2.13

## Layer1: I/O (Input/Output)
**Propósito**: Captura y entrega de datos (audio, texto, imagen)

**Componentes**:
- `true_fullduplex.py`: Multiprocessing I/O
- `input_thread.py`: Captura + STT + Emotion
- `output_thread.py`: LLM + TTS + Playback
- `vosk_streaming.py`: STT con Vosk
- `audio_emotion_lite.py`: Detección emocional
- `lora_router.py`: Routing dinámico de modelos
- `sherpa_vad.py`: Voice Activity Detection

## Layer2: Memory
**Propósito**: Persistencia de contexto y tono

**Componentes**:
- `tone_memory.py`: Memoria de tono emocional

## Layer3: Fluidity
**Propósito**: Transiciones suaves y coordinación

**Componentes**:
- `sherpa_coordinator.py`: Coordinador de VAD
- `tone_bridge.py`: Puente entre Layer1 y Layer2
```

**LOC Estimado**: +150  
**Tiempo**: 2 horas

---

#### Tarea 2.2: Integrar Layer1 con Graph
**Archivo**: `core/graph.py`

**Modificar nodos para usar Layer1**:
```python
# Importar Layer1
from core.layer1_io.audio_emotion_lite import detect_emotion

def classify_intent_node(state: State):
    # Detectar emoción del input (si es audio)
    if state.get("input_type") == "audio":
        emotion_scores = detect_emotion(state["audio_path"])
        state["emotion"] = emotion_scores
    
    # Clasificación TRM como antes
    scores = trm_router.invoke(state["input"])
    return {"hard": scores["hard"], "soft": scores["soft"]}
```

**LOC Estimado**: +60  
**Tiempo**: 3 horas

---

#### Tarea 2.3: Integrar Layer2 con MCP
**Archivo**: `core/mcp.py`

**Usar tone_memory para ajustar α/β**:
```python
from core.layer2_memory.tone_memory import ToneMemory

class MCP:
    def __init__(self):
        # ... código existente ...
        self.tone_memory = ToneMemory()
    
    def compute_weights(self, scores: dict, context: str) -> tuple:
        # ... cálculo base de α/β ...
        
        # Ajustar según memoria de tono
        historical_tone = self.tone_memory.get_recent_tone()
        if historical_tone:
            # Si usuario está frustrado, aumentar β (soft)
            if historical_tone["frustration"] > 0.7:
                beta = min(beta + 0.1, 1.0)
                alpha = 1.0 - beta
        
        return alpha, beta
```

**LOC Estimado**: +40  
**Tiempo**: 2 horas

---

#### Tarea 2.4: Activar Layer3 Fluidity
**Archivo**: `core/graph.py`

**Usar tone_bridge para transiciones suaves**:
```python
from core.layer3_fluidity.tone_bridge import ToneBridge

tone_bridge = ToneBridge()

def modulate_soft_node(state: State):
    # ... modulación existente ...
    
    # Aplicar transición suave de tono
    if state.get("previous_tone"):
        smoothed_tone = tone_bridge.smooth_transition(
            previous_tone=state["previous_tone"],
            current_tone=state["beta"]
        )
        state["beta"] = smoothed_tone
    
    return state
```

**LOC Estimado**: +30  
**Tiempo**: 2 horas

---

#### Tarea 2.5: Tests de Integración de Layers
**Archivo**: `tests/test_layer_integration.py` (NUEVO)

**Tests a crear**:
```python
def test_layer1_emotion_detection():
    """Verifica que Layer1 detecta emoción del audio"""
    # TODO

def test_layer2_tone_memory_adjusts_mcp():
    """Verifica que Layer2 ajusta α/β según historial"""
    # TODO

def test_layer3_smooth_transitions():
    """Verifica transiciones suaves de tono"""
    # TODO
```

**LOC Estimado**: +250  
**Tests**: 10-12 tests  
**Tiempo**: 4 horas

---

### Entregables Fase 2
- [x] `docs/LAYER_ARCHITECTURE.md` completo
- [x] Layer1 integrado en graph (emotion detection)
- [x] Layer2 integrado con MCP (tone memory)
- [x] Layer3 activado (tone bridge)
- [x] Tests de integración (10-12 tests)

**Criterio de Éxito**:
- ✅ Layers funcionan juntos sin errores
- ✅ Emoción detectada ajusta routing MCP
- ✅ Memoria de tono influye en respuestas
- ✅ Transiciones de tono son suaves

**Tiempo Total Fase 2**: **15-20 horas** (2-3 días)

---

## ⚠️ PRE-REQUISITO CRÍTICO: llama-cpp-bin Wrapper (ANTES de FASE 3)

### 🚨 Cambio Estratégico en la Arquitectura

**Decisión del usuario**: Todo el código debe usar el wrapper personalizado `llama-cpp-bin`, NO `llama-cpp-python` directamente.

**Impacto**: Antes de implementar sandbox de skills (FASE 3), debemos:
1. Crear `core/llama_cli_wrapper.py` que envuelva llamadas al binario `llama-cli`
2. Refactorizar `model_pool.py` para usar el wrapper
3. Validar NO regresión de latencia/RAM
4. Seguir patrones LangChain (StateGraph, no spaghetti code)

**Razón**: 
- `llama-cpp-bin` es un contenedor Docker con binarios compilados (AVX2/AVX512/NEON)
- Optimizado, comprimido (UPX), con SHA-256 checksums + GPG
- Tamaño: ~18 MB vs ~100-200 MB de Python wheels
- Multi-arch: amd64 + arm64 nativos
- Auditable y versionado

---

### Tarea PRE-3.0: Implementar LlamaCLIWrapper
**Archivo**: `core/llama_cli_wrapper.py` (NUEVO)

**Clase principal**: `LlamaCLIWrapper`

**Características**:
- ✅ Llama a `llama-cli` del contenedor Docker `ghcr.io/iagenerativa/llama-cpp-bin:2.16-rc`
- ✅ Interfaz compatible con `llama_cpp.Llama` (método `__call__()`, `create_chat_completion()`)
- ✅ Timeout dinámico según n_ctx (Risk #5): `timeout = 10s + (n_ctx / 1024) * 10s`
- ✅ Fallback automático a `llama-cpp-python` si Docker no disponible
- ✅ Extracción inteligente de respuesta del stdout de `llama-cli`
- ✅ Soporte para chat completion (formato OpenAI-compatible)

**Comando Docker usado internamente**:
```bash
docker run --rm \
    -v /path/to/models:/models:ro \
    ghcr.io/iagenerativa/llama-cpp-bin:2.16-rc \
    llama-cli \
        --model /models/solar.gguf \
        --ctx-size 2048 \
        --threads 6 \
        --temp 0.7 \
        --top-p 0.95 \
        --n-predict 150 \
        --prompt "texto"
```

**LOC Estimado**: +350  
**Tests**: `tests/test_llama_cli_wrapper.py` (+200 LOC)  
**Tiempo**: 2-3 horas

**Diseño completo**: `docs/LLAMA_CLI_WRAPPER_DESIGN.md` ✅ CREADO

---

### Tarea PRE-3.0.1: Refactorizar model_pool.py
**Archivo**: `core/model_pool.py`

**Modificación de `_load_gguf_cpu()`**:
```python
def _load_gguf_cpu(
    self,
    model_cfg: Dict[str, Any],
    context_length: Optional[int] = None,
    prefetch: bool = False
) -> Any:
    """
    Carga GGUF con llama-cli wrapper (v2.14+)
    
    Cambios v2.14:
    - Usa LlamaCLIWrapper en lugar de llama_cpp.Llama
    - Fallback automático si Docker no disponible
    """
    from core.llama_cli_wrapper import LlamaCLIWrapper
    
    # ... código de descarga GGUF ...
    
    # ✅ NUEVO v2.14: Usar wrapper personalizado
    return LlamaCLIWrapper(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        use_mmap=use_mmap,
        use_mlock=use_mlock,
        verbose=False
    )
```

**LOC Estimado**: +15 (modificación), -10 (remoción)  
**Tiempo**: 1 hora

---

### Tarea PRE-3.0.2: Validación de No-Regresión
**Archivo**: `tests/test_wrapper_regression.py` (NUEVO)

**Tests críticos**:
```python
def test_wrapper_latency_vs_baseline():
    """Verifica que LlamaCLIWrapper NO aumenta latencia"""
    # Baseline con llama-cpp-python
    # Wrapper con LlamaCLIWrapper
    # Assert: latencia_wrapper <= latencia_baseline * 1.1 (10% margen)

def test_wrapper_ram_vs_baseline():
    """Verifica que LlamaCLIWrapper NO aumenta RAM"""
    # Assert: RAM ≤ 12 GB P99

def test_wrapper_response_quality():
    """Verifica que respuestas son equivalentes"""
    # Comparar embeddings de respuestas (similitud > 0.95)
```

**LOC Estimado**: +250  
**Tiempo**: 2 horas

---

### Checklist PRE-REQUISITO

- [x] `docs/LLAMA_CLI_WRAPPER_DESIGN.md` creado
- [ ] `core/llama_cli_wrapper.py` implementado
- [ ] `tests/test_llama_cli_wrapper.py` creados (5 tests básicos)
- [ ] `model_pool.py` refactorizado
- [ ] Tests de regresión ejecutados (latencia, RAM, calidad)
- [ ] Documentación en `copilot-instructions.md` actualizada

**Tiempo Total PRE-REQUISITO**: 4.5 horas

**IMPORTANTE**: Completar este PRE-REQUISITO ANTES de continuar con FASE 3 (sandbox de skills).

---

## 🐳 FASE 3: v2.14 Patch Sandbox (Skills-as-Services) (DÍA 6-9)

### Objetivos
1. Containerizar skills como servicios gRPC
2. Aislamiento de RAM (skills no saturan host)
3. Cold-start <400ms

### Tareas Específicas

#### Tarea 3.1: Diseñar Docker Compose Multi-Service
**Archivo**: `docker-compose.skills.yml` (NUEVO)

**Estructura**:
```yaml
version: '3.8'

services:
  sarai_core:
    build: .
    networks:
      - sarai_internal
    depends_on:
      - skill_programming
      - skill_creative
      - skill_diagnosis
  
  skill_programming:
    build:
      context: ./skills/programming
      dockerfile: Dockerfile.skill
    networks:
      - sarai_internal
    deploy:
      resources:
        limits:
          memory: 2G
    healthcheck:
      test: ["CMD", "grpc_health_probe", "-addr=:50051"]
      interval: 10s
  
  skill_creative:
    build:
      context: ./skills/creative
    networks:
      - sarai_internal
    # ... similar

networks:
  sarai_internal:
    internal: true
```

**LOC Estimado**: +120  
**Tiempo**: 3 horas

---

#### Tarea 3.2: Implementar gRPC Stubs para Skills
**Archivo**: `skills/skill_service.proto` (NUEVO)

**Definición**:
```protobuf
syntax = "proto3";

service SkillService {
  rpc Execute(SkillRequest) returns (SkillResponse);
}

message SkillRequest {
  string query = 1;
  map<string, string> params = 2;
}

message SkillResponse {
  string response = 1;
  float confidence = 2;
  int32 latency_ms = 3;
}
```

**Generar Python stubs**:
```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. skills/skill_service.proto
```

**LOC Estimado**: +50 (proto) + 200 (generated)  
**Tiempo**: 2 horas

---

#### Tarea 3.3: Implementar Skill Server (Programming Example)
**Archivo**: `skills/programming/server.py` (NUEVO)

**Código**:
```python
import grpc
from concurrent import futures
from skill_service_pb2_grpc import SkillServiceServicer, add_SkillServiceServicer_to_server
from skill_service_pb2 import SkillResponse

class ProgrammingSkillServer(SkillServiceServicer):
    def __init__(self):
        # Cargar LFM2 o SOLAR aquí
        pass
    
    def Execute(self, request, context):
        # Aplicar skill config
        from core.skill_configs import PROGRAMMING_SKILL
        
        prompt = PROGRAMMING_SKILL.build_prompt(request.query)
        params = PROGRAMMING_SKILL.get_generation_params()
        
        # Generar con modelo
        response = self.model.generate(prompt, **params)
        
        return SkillResponse(
            response=response,
            confidence=0.95,
            latency_ms=int(...)
        )

if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    add_SkillServiceServicer_to_server(ProgrammingSkillServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
```

**LOC Estimado**: +150 por skill × 7 skills = 1,050  
**Tiempo**: 8-10 horas (paralelizable)

---

#### Tarea 3.4: Modificar MCP para Llamar Skills vía gRPC
**Archivo**: `core/mcp.py`

**Modificar detect_and_apply_skill**:
```python
def detect_and_apply_skill(query: str, model_name: str = "solar") -> Optional[Dict[str, Any]]:
    skill = match_skill_by_keywords(query)
    
    if skill:
        # NUEVO: Llamar skill containerizado vía gRPC
        import grpc
        from skills import skill_service_pb2_grpc, skill_service_pb2
        
        # Conectar al servicio apropiado
        skill_port = {
            "programming": 50051,
            "creative": 50052,
            "diagnosis": 50053,
            # ...
        }[skill.name]
        
        channel = grpc.insecure_channel(f'skill_{skill.name}:{skill_port}')
        stub = skill_service_pb2_grpc.SkillServiceStub(channel)
        
        # Ejecutar skill remotamente
        request = skill_service_pb2.SkillRequest(
            query=query,
            params={}
        )
        
        response = stub.Execute(request, timeout=5.0)
        
        return {
            "skill_name": skill.name,
            "response": response.response,
            "latency_ms": response.latency_ms,
            "confidence": response.confidence
        }
    
    return None
```

**LOC Estimado**: +80  
**Tiempo**: 3 horas

---

#### Tarea 3.5: Tests de Skills-as-Services
**Archivo**: `tests/test_skills_as_services.py` (NUEVO)

**Tests**:
```python
def test_programming_skill_grpc_call():
    """Verifica llamada gRPC a skill programming"""
    # TODO

def test_skill_cold_start_latency():
    """Verifica cold-start <400ms"""
    # TODO

def test_skill_ram_isolation():
    """Verifica que skill no satura host"""
    # TODO
```

**LOC Estimado**: +300  
**Tests**: 15 tests  
**Tiempo**: 5 horas

---

### Entregables Fase 3
- [x] `docker-compose.skills.yml` con 7 servicios
- [x] gRPC stubs generados
- [x] 7 skill servers implementados
- [x] MCP modificado para llamar skills vía gRPC
- [x] Tests de servicios (15 tests)
- [x] Documentación `docs/SKILLS_AS_SERVICES.md`

**Criterio de Éxito**:
- ✅ Skills responden vía gRPC <400ms cold-start
- ✅ RAM host NO aumenta al usar skills
- ✅ 7 skills funcionan en containers

**Tiempo Total Fase 3**: **25-30 horas** (3-4 días)

---

## 🔐 FASE 4: v2.15 Sentience (GPG Signer + Auditoría) (DÍA 10-12)

### Objetivos
1. Implementar `core/gpg_signer.py`
2. Firmar todas las decisiones del MCP
3. Auditoría HMAC completa

### Tareas Específicas

#### Tarea 4.1: Implementar GPG Signer
**Archivo**: `core/gpg_signer.py` (NUEVO)

**Código**:
```python
import gnupg
import hashlib
import json
from datetime import datetime
from typing import Dict, Any

class GPGSigner:
    """
    Firma decisiones críticas con GPG para trazabilidad
    """
    
    def __init__(self, gpg_home: str = "~/.gnupg"):
        self.gpg = gnupg.GPG(gnupghome=gpg_home)
    
    def sign_decision(self, decision: Dict[str, Any]) -> str:
        """
        Firma una decisión del MCP con GPG
        
        Returns: Signature (PGP armored)
        """
        decision_json = json.dumps(decision, sort_keys=True)
        signature = self.gpg.sign(decision_json)
        return str(signature)
    
    def verify_decision(self, decision: Dict[str, Any], signature: str) -> bool:
        """Verifica firma de una decisión"""
        decision_json = json.dumps(decision, sort_keys=True)
        verified = self.gpg.verify(signature)
        return verified.valid
```

**LOC Estimado**: +120  
**Tiempo**: 3 horas

---

#### Tarea 4.2: Integrar GPG Signer con MCP
**Archivo**: `core/mcp.py`

**Firmar decisiones de routing**:
```python
from core.gpg_signer import GPGSigner

class MCP:
    def __init__(self):
        # ... código existente ...
        self.gpg_signer = GPGSigner()
    
    def compute_weights(self, scores: dict, context: str) -> tuple:
        alpha, beta = self._compute_weights_internal(scores, context)
        
        # Firmar decisión
        decision = {
            "timestamp": datetime.now().isoformat(),
            "scores": scores,
            "alpha": alpha,
            "beta": beta,
            "context_hash": hashlib.sha256(context.encode()).hexdigest()
        }
        
        signature = self.gpg_signer.sign_decision(decision)
        
        # Guardar en log de auditoría
        self._log_signed_decision(decision, signature)
        
        return alpha, beta
```

**LOC Estimado**: +50  
**Tiempo**: 2 horas

---

#### Tarea 4.3: Auditoría HMAC Completa
**Archivo**: `core/web_audit.py` (ya existe, extender)

**Agregar verificación de cadena de confianza**:
```python
def verify_decision_chain(log_file: str) -> bool:
    """
    Verifica que todas las decisiones en el log:
    1. Tienen firma GPG válida
    2. Tienen HMAC correcto
    3. No hay gaps temporales
    """
    # TODO: Implementar verificación completa
    pass
```

**LOC Estimado**: +80  
**Tiempo**: 3 horas

---

#### Tarea 4.4: Tests de GPG Signer
**Archivo**: `tests/test_gpg_signer.py` (NUEVO)

**Tests**:
```python
def test_sign_decision():
    """Verifica que decisión se firma correctamente"""
    # TODO

def test_verify_valid_signature():
    """Verifica firma válida"""
    # TODO

def test_reject_tampered_decision():
    """Verifica rechazo de decisión alterada"""
    # TODO
```

**LOC Estimado**: +200  
**Tests**: 10 tests  
**Tiempo**: 4 horas

---

### Entregables Fase 4
- [x] `core/gpg_signer.py` implementado
- [x] MCP firmando todas las decisiones
- [x] Auditoría HMAC extendida
- [x] Tests de GPG (10 tests)
- [x] Documentación `docs/GPG_SIGNING.md`

**Criterio de Éxito**:
- ✅ Todas las decisiones firmadas con GPG
- ✅ Verificación de cadena funcional
- ✅ 0 decisiones sin firma en logs

**Tiempo Total Fase 4**: **15-18 horas** (2-3 días)

---

## 🧠 FASE 5: v2.16 Omni Loop (Reflexive AGI) (DÍA 13-17)

### Objetivos
1. Draft LLM para iteraciones rápidas
2. Image Preprocessor containerizado
3. LoRA nightly training
4. Reflexive prompting

### Tareas Específicas

#### Tarea 5.1: Implementar Draft LLM Container
**Archivo**: `skills/draft_llm/server.py` (NUEVO)

**Propósito**: LLM rápido (LFM2) para drafts que SOLAR refina

**Código**:
```python
class DraftLLMServer(SkillServiceServicer):
    """
    Draft LLM: Genera respuestas rápidas (0.5s) que SOLAR refina
    """
    
    def __init__(self):
        # Cargar LFM2 (1.2B, rápido)
        self.model = load_lfm2()
    
    def Execute(self, request, context):
        # Generar draft rápido
        draft = self.model.generate(
            request.query,
            temperature=0.8,  # Alta para creatividad
            max_tokens=512    # Corto para velocidad
        )
        
        return SkillResponse(response=draft, latency_ms=500)
```

**LOC Estimado**: +150  
**Tiempo**: 3 horas

---

#### Tarea 5.2: Implementar Omni-Loop Engine
**Archivo**: `core/omni_loop.py` (NUEVO)

**Lógica de reflexión**:
```python
class OmniLoop:
    """
    Motor de reflexión: Draft → Critique → Refine → Repeat
    """
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.draft_llm = DraftLLMClient()
        self.expert_llm = ExpertClient()
    
    def execute(self, query: str) -> str:
        """
        Ejecuta loop reflexivo:
        1. Draft LLM genera respuesta inicial (0.5s)
        2. Expert LLM critica draft (6s)
        3. Si calidad <0.8, genera nuevo draft mejorado
        4. Repite hasta max_iterations o calidad >0.8
        """
        draft = self.draft_llm.generate(query)
        
        for i in range(self.max_iterations):
            # Critique draft
            critique = self.expert_llm.critique(draft, query)
            
            if critique["quality"] > 0.8:
                # Draft aceptable, refinar una vez más
                return self.expert_llm.refine(draft, critique)
            
            # Draft inaceptable, regenerar
            draft = self.draft_llm.generate_improved(query, critique)
        
        # Max iterations alcanzado, devolver mejor intento
        return self.expert_llm.refine(draft, critique)
```

**LOC Estimado**: +250  
**Tiempo**: 5 horas

---

#### Tarea 5.3: Image Preprocessor Containerizado
**Archivo**: `skills/image_preprocessor/server.py` (NUEVO)

**Propósito**: Procesar imágenes fuera del host (0MB RAM host)

**Código**:
```python
import cv2
import numpy as np
from PIL import Image

class ImagePreprocessorServer(SkillServiceServicer):
    """
    Preprocesa imágenes: resize, normalize, WebP encoding
    """
    
    def Execute(self, request, context):
        # Recibir imagen en base64
        image_bytes = base64.b64decode(request.image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize
        image = image.resize((512, 512))
        
        # Convertir a WebP (compresión)
        output = io.BytesIO()
        image.save(output, format='WEBP', quality=85)
        
        # Perceptual hash para cache
        phash = imagehash.phash(image)
        
        return ImageResponse(
            image_data=base64.b64encode(output.getvalue()),
            phash=str(phash)
        )
```

**LOC Estimado**: +180  
**Tiempo**: 4 horas

---

#### Tarea 5.4: LoRA Nightly Training
**Archivo**: `scripts/lora_nightly.py` (NUEVO)

**Propósito**: Fine-tune SOLAR con feedback del día

**Código**:
```python
#!/usr/bin/env python3
"""
LoRA Nightly Training

Ejecuta cada 24h vía cron:
1. Lee logs de feedback del día
2. Filtra interacciones con feedback positivo
3. Fine-tune SOLAR con LoRA
4. Guarda checkpoint
5. Swap atómico (sin downtime)
"""

import torch
from peft import get_peft_model, LoraConfig

def train_lora_from_feedback(log_file: str):
    # Cargar SOLAR base
    model = load_solar()
    
    # Configurar LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05
    )
    
    model = get_peft_model(model, lora_config)
    
    # Entrenar con feedback positivo
    dataset = load_positive_feedback(log_file)
    train(model, dataset, epochs=3)
    
    # Guardar checkpoint
    model.save_pretrained("state/lora_checkpoint_new.pt")
    
    # Señal para swap atómico
    Path("state/lora_ready.flag").touch()

if __name__ == '__main__':
    train_lora_from_feedback("logs/feedback_log.jsonl")
```

**LOC Estimado**: +300  
**Tiempo**: 6 horas

---

#### Tarea 5.5: Tests Omni-Loop
**Archivo**: `tests/test_omni_loop.py` (NUEVO)

**Tests**:
```python
def test_omni_loop_improves_quality():
    """Verifica que loop mejora calidad de respuesta"""
    # TODO

def test_draft_llm_latency_under_500ms():
    """Verifica latencia draft <500ms"""
    # TODO

def test_image_preprocessor_zero_ram_host():
    """Verifica que preprocessor no usa RAM del host"""
    # TODO
```

**LOC Estimado**: +400  
**Tests**: 15 tests  
**Tiempo**: 6 horas

---

### Entregables Fase 5
- [x] Draft LLM container implementado
- [x] Omni-Loop engine funcional
- [x] Image preprocessor containerizado
- [x] LoRA nightly training automatizado
- [x] Tests (15 tests)
- [x] Documentación `docs/OMNI_LOOP.md`

**Criterio de Éxito**:
- ✅ Loop reflexivo funciona en <8s (draft 0.5s + expert 6s + critique 1.5s)
- ✅ Auto-corrección >70%
- ✅ Image preprocessor 0MB RAM host
- ✅ LoRA training sin downtime

**Tiempo Total Fase 5**: **30-35 horas** (4-5 días)

---

## ✅ FASE 6: v2.17 Threading - SKIP

**Razón**: v2.18 ya implementa multiprocessing (superior)

**Acción**: Mantener `orchestrator.py` como legacy fallback

---

## ✅ FASE 7: v2.18 Full-Duplex - VALIDACIÓN FINAL (DÍA 18-20)

### Estado Actual
- ✅ Implementado 100%
- ✅ Tests pasando (20+)

### Tareas de Validación

#### Tarea 7.1: Tests de Integración Completa
**Archivo**: `tests/test_v2.18_integration.py` (NUEVO)

**Tests end-to-end**:
```python
def test_full_duplex_with_skills():
    """Verifica full-duplex + skills juntos"""
    # TODO

def test_full_duplex_with_layers():
    """Verifica full-duplex + layers juntos"""
    # TODO

def test_full_duplex_with_omni_loop():
    """Verifica full-duplex + omni-loop juntos"""
    # TODO
```

**LOC Estimado**: +300  
**Tests**: 10 tests  
**Tiempo**: 5 horas

---

#### Tarea 7.2: Benchmarking Final
**Archivo**: `scripts/benchmark_v2.18.py` (NUEVO)

**KPIs a validar**:
```python
# RAM P99 ≤ 12 GB
# Latencia P50 ≤ 20s (normal), ≤7.9s (omni-loop)
# WER < 2.5%
# MOS > 4.0
# Auto-corrección > 70%
```

**Tiempo**: 4 horas

---

#### Tarea 7.3: Documentación Final
**Archivo**: `README_v2.18_FINAL.md` (NUEVO)

**Contenido**:
- Guía de instalación completa
- Arquitectura final con todos los componentes
- KPIs alcanzados
- Roadmap futuro (v2.19+)

**Tiempo**: 3 horas

---

### Entregables Fase 7
- [x] Tests de integración completa (10 tests)
- [x] Benchmarks finales validados
- [x] Documentación final
- [x] Release notes v2.18

**Criterio de Éxito**:
- ✅ TODOS los KPIs alcanzados
- ✅ 0 regresiones vs versiones anteriores
- ✅ Documentación completa

**Tiempo Total Fase 7**: **12-15 horas** (2 días)

---

## 📊 Resumen de Timeline

| Fase | Versión | Días | LOC Estimado | Tests |
|------|---------|------|--------------|-------|
| 1 | v2.12 Integration | 1-2 | +280 | 8-10 |
| 2 | v2.13 Layers | 2-3 | +530 | 10-12 |
| 3 | v2.14 Patch Sandbox | 3-4 | +1,900 | 15 |
| 4 | v2.15 Sentience | 2-3 | +450 | 10 |
| 5 | v2.16 Omni Loop | 4-5 | +1,280 | 15 |
| 6 | v2.17 Threading | SKIP | 0 | 0 |
| 7 | v2.18 Validación | 2 | +600 | 10 |
| **TOTAL** | **v2.12 → v2.18** | **15-20** | **~5,040** | **68-72** |

---

## 🎯 Próximos Pasos INMEDIATOS

**AHORA** (31 Oct, 17:00):
1. ✅ Plan maestro creado (este documento)
2. ⏳ **DECISIÓN**: ¿Empezar con Fase 1 (v2.12 Integration)?

**HOY** (31 Oct, tarde):
- Iniciar Fase 1, Tarea 1.1: Modificar `core/graph.py`

**MAÑANA** (1 Nov):
- Completar Fase 1 (v2.12 Integration)
- Tests end-to-end

**Esta semana** (1-7 Nov):
- Fases 1-3 completas (v2.12, v2.13, v2.14)

**Próxima semana** (8-15 Nov):
- Fases 4-7 completas (v2.15, v2.16, v2.18 validación)

---

## ❓ Decisión Requerida

**¿Proceder con Fase 1 (v2.12 Integration) ahora?**

Si confirmas, empiezo con **Tarea 1.1: Modificar `core/graph.py` para Skill Detection**.
