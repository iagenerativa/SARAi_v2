# SARAi v2.12 "Omni-Sentinel MoE" - Implementation Plan

## 🎯 Executive Summary

**Version**: v2.12.0  
**Codename**: Omni-Sentinel MoE  
**Milestone**: M4 - Mixture of Experts with Skill Specialization  
**Timeline**: 14 días (Nov 8 - Nov 21, 2025)  
**Total LOC**: ~2,850 líneas (1,850 producción + 1,000 tests)  

### Vision

> _"SARAi no sólo responde: especializa, valida, audita y empatiza, convirtiendo cada skill en un módulo trazable, seguro y reutilizable."_

**Problema**: SARAi v2.11 tiene un modelo híbrido SOLAR+LFM2 efectivo pero rígido. No hay especialización profunda por dominios (código, finanzas, diagnóstico).

**Solución v2.12**: Sistema MoE real con **6 skills modulares** auto-contenidos, cada uno con:
- ✅ Pydantic schema estructurado (type-safe output)
- ✅ TRM-Router multi-label (6 cabezas especializadas, <100ms latencia)
- ✅ Auditoría SHA-256 por skill
- ✅ Empathy layer con Omni-3B (convierte output técnico → voz empática)
- ✅ Sentinel mode compatible (skills deshabilitados en safe mode)

**¿Por qué NO usamos Oracle Tier con LLM?**
- ❌ **Inviable en CPU**: Ejecutar un LLM grande como "oráculo" de routing añadiría 10-20s de latencia inicial
- ❌ **RAM insostenible**: Cargar SOLAR solo para decidir qué hacer consumiría 4.8-6GB antes del skill real
- ✅ **TRM-Router es suficiente**: 7M params, <100ms, <100MB RAM, 90%+ accuracy esperada
- ✅ **Filosofía CPU-first**: Clasificador ultraligero + LLM solo cuando se necesita (en el skill)

---

## 📊 KPIs Target v2.12

| KPI | v2.11 Actual | v2.12 Target | Δ | Estado |
|-----|--------------|--------------|---|--------|
| RAM P99 | 10.8 GB | ≤ 10.5 GB | -0.3 GB | ✅ |
| Latency P50 (Skill) | N/A | ≤ 22 s | NEW | ✅ |
| Routing Accuracy | 87% (2-way) | ≥ 90% (8-way) | +3% | 🎯 |
| Skills Supported | 0 | 6+ | NEW | ✅ |
| Log Integrity | 100% (web+voice) | 100% (web+voice+skills) | - | ✅ |
| Structured Output | 0% | 100% (Pydantic) | NEW | ✅ |
| Empathy Layer | Voice only | Voice + Skills | NEW | ✅ |

**Mantra v2.12**:  
_"Cada skill es un experto certificado: entrada validada, salida estructurada, ejecución auditada."_

---

## 🏗️ System Architecture

### Before v2.12 (Rigid Hybrid)

```
Input → TRM (2-way: hard/soft) → MCP (α/β) → SOLAR/LFM2 → Response
```

**Limitaciones**:
- Solo 2 rutas (técnico vs emocional)
- Sin especialización por dominio
- Output de texto libre (no estructurado)
- No auditable por skill

### After v2.12 (Modular MoE)

```
Input → TRM-Router (8-way: hard, soft, 6 skills)
         ↓
    MCP + Sentinel Check
         ↓
    ┌────┴─────┬────────┬─────────┬────────┬─────────┬─────────┐
    ↓          ↓        ↓         ↓        ↓         ↓         ↓
Programming Diagnosis Finance Logic  Creative Reasoning Fallback
 (Skill)    (Skill)   (Skill)  (Skill)  (Skill)  (Skill)  (Tiny)
    ↓          ↓        ↓         ↓        ↓         ↓         ↓
    └──────────┴────────┴─────────┴────────┴─────────┴─────────┘
                              ↓
                     skill.execute(state)
                              ↓
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
       Model Selection    Model Loading   Generation
       (by skill logic)   (ModelPool)     (LLM inference)
       
       Example: Programming Skill
       - hard > 0.7 → expert_short (SOLAR n_ctx=512)
       - else → tiny (LFM2-1.2B)
              ↓               ↓               ↓
              └───────────────┴───────────────┘
                              ↓
                      Structured Output (Pydantic)
                              ↓
                      Empathy Layer (Omni-3B)
                              ↓
                   Audit Trail (SHA-256 per skill)
                              ↓
                          Response
```

**Beneficios v2.12**:
- ✅ 6 skills especializados (extensibles a 10+)
- ✅ TRM-Router Multi-Skill: <100ms, <100MB RAM, 90%+ accuracy
- ✅ Output Pydantic (type-safe, validado)
- ✅ Empathy layer para voz natural
- ✅ Auditoría granular SHA-256 por skill execution
- ✅ RAM controlado (+0.1-0.3 GB/skill activo, avg 2.5GB)

---

## 📁 Directory Structure

```
sarai/
├── skills/                        # NEW v2.12
│   ├── __init__.py
│   ├── base_skill.py              # Abstract base class
│   ├── programming_skill.py       # Código, debug, tests
│   ├── diagnosis_skill.py         # Troubleshooting, análisis
│   ├── finance_skill.py           # Presupuestos, ROI, inversión
│   ├── logic_skill.py             # Razonamiento lógico, if-then
│   ├── creative_skill.py          # Historias, poemas, diseño
│   └── reasoning_skill.py         # Análisis, estrategia, planificación
├── core/
│   ├── trm_router.py              # EXTENDED: Multi-skill classification (8 heads)
│   ├── graph.py                   # MODIFIED: route_to_skill()
│   ├── audit.py                   # EXTENDED: log_skill_execution()
│   └── orchestrator.py            # NEW: Skill orchestration + model selection
├── agents/
│   └── empathy_layer.py           # NEW: Omni-3B wrapper for skills
tests/
├── test_base_skill.py             # NEW: Base class tests
├── test_programming_skill.py      # NEW: Programming skill tests
├── test_diagnosis_skill.py        # NEW: Diagnosis skill tests
├── test_finance_skill.py          # NEW: Finance skill tests
├── test_logic_skill.py            # NEW: Logic skill tests
├── test_creative_skill.py         # NEW: Creative skill tests
├── test_reasoning_skill.py        # NEW: Reasoning skill tests
├── test_skill_routing.py          # NEW: TRM multi-skill routing tests
├── test_skill_orchestration.py   # NEW: Integration tests
└── test_skill_audit.py            # NEW: Audit trail tests
```

**Total Files**: 17 nuevos archivos (8 skills + 9 tests)

---

## 🧩 Component Design

### 1. Base Skill Class

**File**: `sarai/skills/base_skill.py`

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional

class SkillOutput(BaseModel):
    """Base schema for all skill outputs"""
    skill_name: str
    confidence: float  # 0.0 - 1.0
    result: dict  # Skill-specific structured data
    metadata: dict = {}

class BaseSkill(ABC):
    """
    Abstract base class for all SARAi skills
    
    CRITICAL: All skills MUST implement:
    - execute(): Main processing logic
    - get_schema(): Pydantic schema for structured output
    - can_handle(): Routing threshold logic
    """
    
    def __init__(self, name: str, description: str, keywords: list[str]):
        self.name = name
        self.description = description
        self.keywords = keywords
        self.temperature = 0.3  # Default conservative
        self.min_p = 0.15       # Nucleus sampling
    
    @abstractmethod
    def execute(self, query: str, context: dict, model) -> SkillOutput:
        """
        Execute skill with structured output
        
        Args:
            query: User input
            context: Execution context (history, state, etc.)
            model: LLM instance (expert_short, expert_long, tiny)
        
        Returns:
            SkillOutput: Pydantic-validated result
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> type[BaseModel]:
        """
        Return Pydantic schema for this skill's output
        
        Used by LLM for structured generation via model_json_schema()
        """
        pass
    
    def can_handle(self, scores: dict) -> bool:
        """
        Check if skill can handle query based on TRM scores
        
        Default: skill_score > 0.4 (MCP-learned threshold)
        """
        return scores.get(self.name, 0.0) > 0.4
    
    def prepare_prompt(self, query: str, schema: type[BaseModel]) -> str:
        """
        Generate structured prompt with Pydantic schema
        
        CRITICAL: Use model_json_schema() to avoid quote escaping
        """
        schema_json = schema.model_json_schema()
        
        return f"""You are a {self.name} expert in SARAi v2.12.

**Input Query**: {query}

**Your Task**: Analyze the query and provide a structured response following this exact JSON schema:

```json
{schema_json}
```

**Requirements**:
- All fields are mandatory unless marked Optional
- Confidence must be 0.0-1.0
- Include detailed reasoning in metadata
- Output ONLY valid JSON, no markdown wrappers

**Output**:"""
```

**Beneficios**:
- ✅ Type-safe: Pydantic valida output
- ✅ Extensible: Nuevos skills heredan lógica común
- ✅ Prompts consistentes: `model_json_schema()` evita escaping
- ✅ MCP-compatible: `can_handle()` usa thresholds aprendidos

---

### 2. TRM-Router Multi-Skill (The Right Approach for SARAi)

**File**: `sarai/core/trm_router.py` (EXTENDED)

**Why TRM-Router Instead of "Oracle Tier with LLM"?**

Some MoE systems use a large LLM as an "oracle" at the beginning of the pipeline to analyze, refine, or decompose queries before routing. **This is NOT suitable for SARAi v2.12**:

#### ❌ Oracle Tier Approach (Rejected for SARAi)

**Concept**: Use a powerful LLM (e.g., SOLAR-10.7B) as initial preprocessor/router.

**Problems**:
1. **Latency catastrophic**: +10-20s overhead BEFORE the real skill even starts
2. **RAM unsustainable**: Loading SOLAR (4.8-6GB) just to decide routing wastes precious RAM budget
3. **CPU bottleneck**: Running inference on 10.7B model for every query is overkill on CPU
4. **Against SARAi philosophy**: "CPU-first, low-RAM, lightweight"

#### ✅ TRM-Router Multi-Skill (SARAi's Approach)

**Concept**: Ultra-lightweight classifier (7M params) for intelligent routing.

**Advantages**:
1. **Latency**: <100ms routing overhead (vs 10-20s with LLM oracle)
2. **RAM**: <100MB (vs 4.8-6GB with LLM oracle)
3. **Accuracy**: 90%+ expected (sufficient for skill routing)
4. **Scalability**: Can handle 1000+ requests/min on CPU

**Implementation**:

```python
# sarai/core/trm_router.py - TRM-Router Multi-Skill (v2.12)

SKILL_DOMAINS = {
    "programming": [
        "código", "función", "debug", "test", "python", "javascript",
        "clase", "método", "algoritmo", "sintaxis", "error"
    ],
    "diagnosis": [
        "problema", "error", "fallo", "síntoma", "diagnóstico",
        "no funciona", "roto", "crashea", "arreglar", "solucionar"
    ],
    "finance": [
        "presupuesto", "inversión", "costo", "ingreso", "roi",
        "gasto", "beneficio", "ahorro", "deuda", "precio"
    ],
    "logic": [
        "si...entonces", "premisa", "conclusión", "deducir", "implica",
        "por lo tanto", "inferir", "demostrar", "prueba", "válido"
    ],
    "creative": [
        "crea", "diseña", "inventa", "historia", "poema",
        "imagina", "genera", "escribe", "compón", "narra"
    ],
    "reasoning": [
        "analiza", "evalúa", "compara", "estrategia", "planifica",
        "decide", "prioriza", "optimiza", "razona", "justifica"
    ]
}

class TRMRouterMultiSkill(nn.Module):
    """
    TRM-Router v2.12 with 8 classification heads
    
    Heads:
    - hard, soft (base, inherited from v2.11)
    - programming, diagnosis, finance, logic, creative, reasoning (skills)
    
    CRITICAL: 
    - Multi-label (NO softmax), each skill independent
    - <100ms latency, <100MB RAM
    - 90%+ accuracy expected
    """
    
    def __init__(self, d_model=256, K_cycles=3):
        super().__init__()
        self.projection = nn.Linear(2048, d_model)  # EmbeddingGemma → TRM
        self.recursive_layer = TinyRecursiveLayer(d_model, d_model)
        
        # Base heads (v2.11 compatibility)
        self.head_hard = nn.Linear(d_model, 1)
        self.head_soft = nn.Linear(d_model, 1)
        
        # Skill heads (v2.12 NEW)
        self.head_programming = nn.Linear(d_model, 1)
        self.head_diagnosis = nn.Linear(d_model, 1)
        self.head_finance = nn.Linear(d_model, 1)
        self.head_logic = nn.Linear(d_model, 1)
        self.head_creative = nn.Linear(d_model, 1)
        self.head_reasoning = nn.Linear(d_model, 1)
        
        self.K_cycles = K_cycles
    
    def forward(self, x_embedding: torch.Tensor) -> dict:
        """
        Multi-label classification (sigmoid per head, NO softmax)
        
        Why NO softmax?
        - A query can need multiple skills: "Debug this code and explain why"
          → programming: 0.85, diagnosis: 0.75
        - Enables skill composition in future (v2.13)
        
        Latency: ~50-80ms on i7 CPU
        """
        x = self.projection(x_embedding)  # 2048 → 256
        
        # Recursive processing (K=3 cycles)
        y, z = torch.zeros(256), torch.zeros(256)
        for _ in range(self.K_cycles):
            z = self.recursive_layer.f_z(x, y, z)
            y = self.recursive_layer.f_y(y, z)
        
        # Multi-label classification (sigmoid individual)
        return {
            "hard": torch.sigmoid(self.head_hard(y)).item(),
            "soft": torch.sigmoid(self.head_soft(y)).item(),
            "programming": torch.sigmoid(self.head_programming(y)).item(),
            "diagnosis": torch.sigmoid(self.head_diagnosis(y)).item(),
            "finance": torch.sigmoid(self.head_finance(y)).item(),
            "logic": torch.sigmoid(self.head_logic(y)).item(),
            "creative": torch.sigmoid(self.head_creative(y)).item(),
            "reasoning": torch.sigmoid(self.head_reasoning(y)).item()
        }
    
    def keyword_boost(self, input_text: str, scores: dict) -> dict:
        """
        Boost scores based on keyword matching
        
        Used for:
        - Cold-start (model not trained yet)
        - Fallback if TRM confidence is low
        """
        text_lower = input_text.lower()
        
        for skill_name, keywords in SKILL_DOMAINS.items():
            keyword_score = sum(1 for k in keywords if k in text_lower) / len(keywords)
            
            # Boost existing score or create new if absent
            if skill_name in scores:
                scores[skill_name] = min(scores[skill_name] + keyword_score * 0.2, 1.0)
            else:
                scores[skill_name] = keyword_score
        
        return scores
```

**Training Strategy**:
1. **Phase 1 (Cold-start)**: Only keyword matching
2. **Phase 2 (100-500 samples)**: Distillation from SOLAR (generate 5000 synthetic examples)
3. **Phase 3 (500+ samples)**: Fine-tuning with real feedback

**Comparison: TRM-Router vs Oracle Tier**

| Aspect | TRM-Router (SARAi) | Oracle Tier (Rejected) |
|--------|-------------------|------------------------|
| **Latency** | <100ms | +10-20s |
| **RAM** | <100MB | 4.8-6GB |
| **Purpose** | Skill routing | Query preprocessing + routing |
| **Model Size** | 7M params | 10.7B params |
| **CPU Efficiency** | ✅ Optimized | ❌ Bottleneck |
| **Accuracy** | 90%+ (sufficient) | 95%+ (overkill) |
| **Philosophy Fit** | ✅ Perfect | ❌ Against "CPU-first" |

**Conclusion**: TRM-Router is the **pragmatic and efficient** routing system for SARAi's strict resource constraints.

---

### 3. Skills Base Class + Specializations
# TRM-Router v2.12: Multi-label classification with 8 outputs

SKILL_DOMAINS = {
    "programming": [
        "código", "función", "debug", "test", "python", "javascript",
        "código", "clase", "método", "algoritmo", "sintaxis", "error"
    ],
    "diagnosis": [
        "problema", "error", "fallo", "síntoma", "diagnóstico",
        "no funciona", "roto", "crashea", "arreglar", "solucionar"
    ],
    "finance": [
        "presupuesto", "inversión", "costo", "ingreso", "roi",
        "gasto", "beneficio", "ahorro", "deuda", "precio"
    ],
    "logic": [
        "si...entonces", "premisa", "conclusión", "deducir", "implica",
        "por lo tanto", "inferir", "demostrar", "prueba", "válido"
    ],
    "creative": [
        "crea", "diseña", "inventa", "historia", "poema",
        "imagina", "genera", "escribe", "compón", "narra"
    ],
    "reasoning": [
        "analiza", "evalúa", "compara", "estrategia", "planifica",
        "decide", "prioriza", "optimiza", "razona", "justifica"
    ]
}

class TRMRouterMultiSkill(nn.Module):
    """
    TRM-Router v2.12 con 8 cabezas de clasificación:
    - hard, soft (base, heredado de v2.11)
    - programming, diagnosis, finance, logic, creative, reasoning (skills)
    
    CRITICAL: Multi-label (no softmax), umbral dinámico por MCP
    """
    
    def __init__(self, d_model=256, K_cycles=3):
        super().__init__()
        self.projection = nn.Linear(2048, d_model)  # EmbeddingGemma → TRM
        self.recursive_layer = TinyRecursiveLayer(d_model, d_model)
        
        # Base heads (v2.11 compatibility)
        self.head_hard = nn.Linear(d_model, 1)
        self.head_soft = nn.Linear(d_model, 1)
        
        # Skill heads (v2.12 NEW)
        self.head_programming = nn.Linear(d_model, 1)
        self.head_diagnosis = nn.Linear(d_model, 1)
        self.head_finance = nn.Linear(d_model, 1)
        self.head_logic = nn.Linear(d_model, 1)
        self.head_creative = nn.Linear(d_model, 1)
        self.head_reasoning = nn.Linear(d_model, 1)
        
        self.K_cycles = K_cycles
    
    def forward(self, x_embedding: torch.Tensor) -> dict:
        """
        Returns: Dict with 8 scores (sigmoid, NOT softmax)
        
        Multi-label: Una query puede tener score alto en múltiples skills
        Ej: "Escribe tests para esta función Python"
            → programming: 0.85, creative: 0.4, reasoning: 0.3
        """
        x = self.projection(x_embedding)  # 2048 → 256
        
        # Recursive processing (K=3 cycles)
        y, z = torch.zeros(256), torch.zeros(256)
        for _ in range(self.K_cycles):
            z = self.recursive_layer.f_z(x, y, z)
            y = self.recursive_layer.f_y(y, z)
        
        # Multi-label classification (sigmoid individual)
        return {
            "hard": torch.sigmoid(self.head_hard(y)).item(),
            "soft": torch.sigmoid(self.head_soft(y)).item(),
            "programming": torch.sigmoid(self.head_programming(y)).item(),
            "diagnosis": torch.sigmoid(self.head_diagnosis(y)).item(),
            "finance": torch.sigmoid(self.head_finance(y)).item(),
            "logic": torch.sigmoid(self.head_logic(y)).item(),
            "creative": torch.sigmoid(self.head_creative(y)).item(),
            "reasoning": torch.sigmoid(self.head_reasoning(y)).item()
        }
    
    def keyword_boost(self, input_text: str, scores: dict) -> dict:
        """
        Boost scores based on keyword matching
        
        Usado en cold-start (modelo sin entrenar) o como fallback
        """
        text_lower = input_text.lower()
        
        for skill_name, keywords in SKILL_DOMAINS.items():
            keyword_score = sum(1 for k in keywords if k in text_lower) / len(keywords)
            
            # Boost existing score o crear nuevo si está ausente
            if skill_name in scores:
                scores[skill_name] = min(scores[skill_name] + keyword_score * 0.2, 1.0)
            else:
                scores[skill_name] = keyword_score
        
        return scores
```

**Training Strategy**:
1. **Phase 1 (Cold-start)**: Solo keyword matching
2. **Phase 2 (100-500 samples)**: Distilación de SOLAR (generar 5000 ejemplos sintéticos)
3. **Phase 3 (500+ samples)**: Fine-tuning con feedback real

---

### 3. Programming Skill (Example)

**File**: `sarai/skills/programming_skill.py`

```python
from pydantic import BaseModel, Field
from .base_skill import BaseSkill, SkillOutput

class CodeSolution(BaseModel):
    """Structured output for programming skill"""
    language: str = Field(description="Programming language detected/used")
    code: str = Field(description="Solution code")
    explanation: str = Field(description="Step-by-step explanation")
    test_cases: list[str] = Field(description="Suggested test cases")
    complexity: dict = Field(description="Time/space complexity analysis")
    best_practices: list[str] = Field(default=[], description="Code quality tips")

class ProgrammingSkill(BaseSkill):
    """
    Expert in code generation, debugging, testing, and optimization
    
    Specializations:
    - Code generation from natural language
    - Bug diagnosis and fixes
    - Test case generation
    - Code review and refactoring
    - Algorithm complexity analysis
    """
    
    def __init__(self):
        super().__init__(
            name="programming",
            description="Expert in software development and debugging",
            keywords=["código", "función", "debug", "test", "python", "javascript"]
        )
        self.temperature = 0.3  # Low for deterministic code
        self.min_p = 0.15       # Conservative sampling
    
    def get_schema(self) -> type[BaseModel]:
        return CodeSolution
    
    def execute(self, query: str, context: dict, model) -> SkillOutput:
        """
        Execute programming skill with SOLAR expert_short
        
        Flow:
        1. Prepare structured prompt with CodeSolution schema
        2. Generate with temperature=0.3, min_p=0.15
        3. Parse JSON output → Pydantic validation
        4. Return SkillOutput with confidence
        """
        # Step 1: Prepare prompt
        prompt = self.prepare_prompt(query, CodeSolution)
        
        # Step 2: Generate with LLM (SOLAR preferred)
        response = model.generate(
            prompt,
            temperature=self.temperature,
            min_p=self.min_p,
            max_tokens=2048,
            stop=["```", "\n\n\n"]  # Avoid over-generation
        )
        
        # Step 3: Parse and validate
        try:
            import json
            code_solution = CodeSolution.model_validate_json(response)
            confidence = self._calculate_confidence(code_solution, query)
            
            return SkillOutput(
                skill_name=self.name,
                confidence=confidence,
                result=code_solution.model_dump(),
                metadata={
                    "model": model.__class__.__name__,
                    "temperature": self.temperature,
                    "tokens": len(response.split())
                }
            )
        
        except Exception as e:
            # Fallback: Return low-confidence error response
            return SkillOutput(
                skill_name=self.name,
                confidence=0.2,
                result={
                    "language": "unknown",
                    "code": "",
                    "explanation": f"Failed to parse response: {str(e)}",
                    "test_cases": [],
                    "complexity": {},
                    "best_practices": []
                },
                metadata={"error": str(e)}
            )
    
    def _calculate_confidence(self, solution: CodeSolution, query: str) -> float:
        """
        Calculate confidence based on:
        - Code length (min 10 chars)
        - Explanation quality (min 50 chars)
        - Test cases present (bonus +0.1)
        - Complexity analysis present (bonus +0.1)
        """
        score = 0.5  # Base confidence
        
        if len(solution.code) > 10:
            score += 0.2
        
        if len(solution.explanation) > 50:
            score += 0.2
        
        if solution.test_cases:
            score += 0.1
        
        if solution.complexity:
            score += 0.1
        
        return min(score, 1.0)
```

**Similar structure** for:
- `diagnosis_skill.py`
- `finance_skill.py`
- `logic_skill.py`
- `creative_skill.py`
- `reasoning_skill.py`

---

### 4. Orchestrator Integration

**File**: `sarai/core/orchestrator.py` (NEW)

```python
from sarai.skills import get_all_skills
from sarai.core.audit import log_skill_execution, is_sentinel_active
from sarai.core.model_pool import get_model_pool
from sarai.agents.empathy_layer import empathize_output

class SkillOrchestrator:
    """
    Central orchestrator for skill routing and execution
    
    Flow:
    1. TRM-Router → Scores (8-way)
    2. MCP → Thresholds + Sentinel check
    3. Select best skill (score > threshold)
    4. Execute skill → Structured output (Pydantic)
    5. Empathy layer → Natural voice (Omni-3B)
    6. Audit → SHA-256 log
    """
    
    def __init__(self):
        self.skills = get_all_skills()  # Load all registered skills
        self.model_pool = get_model_pool()
    
    def route_and_execute(self, scores: dict, query: str, context: dict) -> dict:
        """
        Main routing logic for skill selection
        
        Returns:
            {
                "response": str,           # Final empathetic response
                "structured_output": dict, # Pydantic-validated result
                "skill_used": str,         # Skill name
                "confidence": float,       # Skill confidence
                "audit_hash": str          # SHA-256 of execution
            }
        """
        # Step 1: Sentinel check
        if is_sentinel_active():
            return {
                "response": "Skills deshabilitados en modo seguro.",
                "structured_output": {},
                "skill_used": "sentinel",
                "confidence": 1.0,
                "audit_hash": None
            }
        
        # Step 2: Select best skill
        skill_candidates = {
            k: v for k, v in scores.items() 
            if k not in ["hard", "soft"] and v > 0.4
        }
        
        if not skill_candidates:
            # Fallback to base SOLAR/LFM2 routing
            return self._fallback_routing(scores, query, context)
        
        best_skill_name = max(skill_candidates, key=skill_candidates.get)
        skill = self.skills[best_skill_name]
        
        # Step 3: Select model (expert_short if hard > 0.7, else tiny)
        model_name = "expert_short" if scores["hard"] > 0.7 else "tiny"
        model = self.model_pool.get(model_name)
        
        # Step 4: Execute skill
        skill_output = skill.execute(query, context, model)
        
        # Step 5: Empathy layer (convert structured → natural voice)
        empathetic_response = empathize_output(
            skill_output=skill_output,
            user_query=query,
            emotion=context.get("detected_emotion", "neutral")
        )
        
        # Step 6: Audit trail
        audit_hash = log_skill_execution(
            skill_name=best_skill_name,
            input_query=query,
            output=skill_output,
            user_id=context.get("user_id", "anonymous")
        )
        
        # Step 7: Release model
        self.model_pool.release(model_name)
        
        return {
            "response": empathetic_response,
            "structured_output": skill_output.result,
            "skill_used": best_skill_name,
            "confidence": skill_output.confidence,
            "audit_hash": audit_hash
        }
    
    def _fallback_routing(self, scores: dict, query: str, context: dict) -> dict:
        """
        Fallback to v2.11 routing (SOLAR/LFM2 hybrid)
        
        Used when no skill qualifies (all scores < 0.4)
        """
        # ... existing v2.11 logic ...
        pass
```

---

### 5. Empathy Layer

**File**: `sarai/agents/empathy_layer.py` (NEW)

```python
from sarai.core.model_pool import get_model_pool

def empathize_output(skill_output, user_query: str, emotion: str = "neutral") -> str:
    """
    Convert structured skill output to empathetic natural voice
    
    Uses Qwen2.5-Omni-3B for voice-friendly responses
    
    Args:
        skill_output: SkillOutput from any skill
        user_query: Original user query
        emotion: Detected emotion (empático, neutral, urgente)
    
    Returns:
        Natural language response with empathy
    """
    model_pool = get_model_pool()
    omni_model = model_pool.get("omni3b")
    
    # Construct empathy prompt
    prompt = f"""You are SARAi, an empathetic AI assistant.

**User asked**: {user_query}

**Structured answer from {skill_output.skill_name} skill**:
```json
{skill_output.model_dump_json(indent=2)}
```

**Your task**: Convert this structured output into a {emotion} natural response.

**Requirements**:
- Maintain all technical accuracy
- Use conversational tone
- Address user directly ("you", "your")
- Keep it concise (max 200 words)
- Match emotion: {emotion}

**Response**:"""
    
    response = omni_model.generate(
        prompt,
        temperature=0.7,  # Higher for natural variation
        max_tokens=300
    )
    
    model_pool.release("omni3b")
    return response.strip()
```

---

### 6. Audit Trail

**File**: `sarai/core/audit.py` (EXTENDED)

```python
import hashlib
import json
from datetime import datetime

def log_skill_execution(skill_name: str, input_query: str, output, user_id: str) -> str:
    """
    Log skill execution with SHA-256 immutability
    
    File structure:
    logs/skills/{skill_name}_{date}.jsonl
    logs/skills/{skill_name}_{date}.jsonl.sha256
    
    Returns:
        SHA-256 hash of the log entry
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "skill": skill_name,
        "input_hash": hashlib.sha256(input_query.encode()).hexdigest(),
        "output_hash": hashlib.sha256(
            json.dumps(output.result, sort_keys=True).encode()
        ).hexdigest(),
        "user_id_hash": hashlib.sha256(user_id.encode()).hexdigest(),
        "confidence": output.confidence,
        "metadata": output.metadata
    }
    
    # Serialize entry
    entry_json = json.dumps(entry, ensure_ascii=False, sort_keys=True)
    
    # Compute SHA-256
    entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()
    
    # Write to log
    date = datetime.now().strftime("%Y-%m-%d")
    log_path = f"logs/skills/{skill_name}_{date}.jsonl"
    hash_path = f"{log_path}.sha256"
    
    with open(log_path, "a") as f:
        f.write(entry_json + "\n")
    
    with open(hash_path, "a") as f:
        f.write(f"{entry_hash}\n")
    
    return entry_hash
```

---

## 📅 Implementation Timeline

### **M1: Base Skill Architecture** (Day 1-2, Nov 8-9)

**Goal**: Crear infraestructura base para skills modulares

**Tasks**:
- ✅ Implementar `BaseSkill` abstract class
- ✅ Implementar `SkillOutput` Pydantic schema
- ✅ Crear `skills/__init__.py` con registry
- ✅ Tests: 5 tests unitarios de base class

**Files**:
- `sarai/skills/base_skill.py` (+150 LOC)
- `sarai/skills/__init__.py` (+50 LOC)
- `tests/test_base_skill.py` (+100 LOC)

**Validation**: `pytest tests/test_base_skill.py -v`

---

### **M2: TRM-Router Multi-Skill** (Day 3-4, Nov 10-11)

**Goal**: Extender TRM-Router a clasificación multi-label (8 cabezas)

**Tasks**:
- ✅ Añadir 6 cabezas de skills a TRM-Router
- ✅ Implementar `keyword_boost()` para cold-start
- ✅ Generar dataset sintético (5000 samples con SOLAR)
- ✅ Entrenar TRM con distilación
- ✅ Tests: 8 tests de routing multi-skill

**Files**:
- `sarai/core/trm_router.py` (+200 LOC modificado)
- `scripts/train_trm_v2_12.py` (+150 LOC nuevo)
- `tests/test_skill_routing.py` (+150 LOC)

**Command**: `python scripts/train_trm_v2_12.py --epochs 50 --samples 5000`

**Validation**: `pytest tests/test_skill_routing.py -v`

---

### **M3: Skills Implementation** (Day 5-8, Nov 12-15)

**Goal**: Implementar 5 skills completos con Pydantic schemas

**Tasks per skill** (repeat 5x):
- ✅ Definir Pydantic schema (ej: `CodeSolution`)
- ✅ Implementar `execute()` con structured generation
- ✅ Implementar `_calculate_confidence()`
- ✅ Tests: 5 tests por skill (25 total)

**Skills**:
1. `programming_skill.py` (+250 LOC)
2. `diagnosis_skill.py` (+220 LOC)
3. `finance_skill.py` (+230 LOC)
4. `logic_skill.py` (+200 LOC)
5. `creative_skill.py` (+210 LOC)

**Total**: ~1,100 LOC producción + 500 LOC tests

**Validation**: `pytest tests/test_*_skill.py -v`

---

### **M4: Orchestrator Integration** (Day 9-10, Nov 16-17)

**Goal**: Integrar skills en LangGraph con empathy layer

**Tasks**:
- ✅ Implementar `SkillOrchestrator` class
- ✅ Integrar en `core/graph.py` (nuevo nodo `execute_skill`)
- ✅ Implementar empathy layer con Omni-3B
- ✅ Routing condicional: skill vs fallback
- ✅ Tests: 10 tests de integración

**Files**:
- `sarai/core/orchestrator.py` (+200 LOC nuevo)
- `sarai/agents/empathy_layer.py` (+100 LOC nuevo)
- `sarai/core/graph.py` (+50 LOC modificado)
- `tests/test_skill_orchestration.py` (+200 LOC)

**Graph Flow**:
```python
workflow.add_node("execute_skill", orchestrator.route_and_execute)
workflow.add_conditional_edges(
    "mcp",
    lambda state: "execute_skill" if has_skill_match(state) else "generate_expert",
    {"execute_skill": "feedback", "generate_expert": "generate_expert"}
)
```

**Validation**: `pytest tests/test_skill_orchestration.py -v`

---

### **M5: Audit & Security** (Day 11, Nov 18)

**Goal**: Implementar auditoría SHA-256 por skill

**Tasks**:
- ✅ Extender `core/audit.py` con `log_skill_execution()`
- ✅ Crear logs separados por skill (`logs/skills/`)
- ✅ Implementar verificación de integridad
- ✅ Tests: 6 tests de auditoría

**Files**:
- `sarai/core/audit.py` (+100 LOC modificado)
- `scripts/verify_skill_audit.py` (+100 LOC nuevo)
- `tests/test_skill_audit.py` (+120 LOC)

**Command**: `python scripts/verify_skill_audit.py --skill programming --date 2025-11-18`

**Validation**: `pytest tests/test_skill_audit.py -v`

---

### **M6: E2E Testing & Validation** (Day 12-13, Nov 19-20)

**Goal**: Tests end-to-end de todos los skills y validación de KPIs

**Tasks**:
- ✅ Tests E2E de 6 skills (20+ escenarios)
- ✅ Benchmark de RAM (P99 ≤ 10.5 GB)
- ✅ Benchmark de latencia (P50 ≤ 22s)
- ✅ Accuracy de routing (≥ 90%)
- ✅ Validación de auditoría (100% integridad)

**Files**:
- `tests/test_e2e_skills.py` (+300 LOC)
- `tests/sarai_bench_v2_12.py` (+100 LOC)

**Command**: `pytest tests/test_e2e_skills.py -v -s`

**Validation**: `python tests/sarai_bench_v2_12.py --validate-kpis`

---

### **M7: Release & Documentation** (Day 14, Nov 21)

**Goal**: Tag v2.12.0, crear release, actualizar documentación

**Tasks**:
- ✅ Tag `v2.12.0`
- ✅ GitHub Release con CHANGELOG
- ✅ Actualizar `README_v2.12.md`
- ✅ Crear **Skill Development Guide**
- ✅ Actualizar `.github/copilot-instructions.md`

**Files**:
- `CHANGELOG.md` (+100 LOC)
- `README_v2.12.md` (+200 LOC)
- `docs/SKILL_DEVELOPMENT_GUIDE.md` (+400 LOC nuevo)
- `.github/copilot-instructions.md` (+150 LOC)

**Commands**:
```bash
git tag v2.12.0
git push origin v2.12.0
gh release create v2.12.0 --title "SARAi v2.12 Omni-Sentinel MoE" --notes-file CHANGELOG.md
```

---

## 🔬 Testing Strategy

### Unit Tests (50+ tests)

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| `test_base_skill.py` | 5 | Base class validation |
| `test_programming_skill.py` | 5 | Code generation, debugging |
| `test_diagnosis_skill.py` | 5 | Problem analysis |
| `test_finance_skill.py` | 5 | Budget, ROI calculations |
| `test_logic_skill.py` | 5 | Logical reasoning |
| `test_creative_skill.py` | 5 | Story, poem generation |
| `test_reasoning_skill.py` | 5 | Strategic analysis |
| `test_skill_routing.py` | 8 | Multi-skill classification |
| `test_skill_audit.py` | 6 | SHA-256 integrity |

**Total**: 49 unit tests

### Integration Tests (10 tests)

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| `test_skill_orchestration.py` | 10 | LangGraph integration, empathy layer |

### E2E Tests (20+ tests)

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| `test_e2e_skills.py` | 20+ | Complete flows: input → skill → empathy → audit |

**Total Tests**: 79+ tests (49 unit + 10 integration + 20 E2E)

---

## 📊 Validation Criteria

### KPI Compliance

```python
# tests/sarai_bench_v2_12.py

def validate_kpis():
    """Validate all v2.12 KPIs"""
    
    results = {
        "ram_p99": measure_ram_p99(),       # Target: ≤ 10.5 GB
        "latency_p50": measure_latency(),   # Target: ≤ 22 s
        "routing_accuracy": measure_accuracy(),  # Target: ≥ 90%
        "log_integrity": verify_all_logs(),      # Target: 100%
        "structured_output": validate_pydantic() # Target: 100%
    }
    
    assert results["ram_p99"] <= 10.5, f"RAM P99 exceeded: {results['ram_p99']:.1f} GB"
    assert results["latency_p50"] <= 22, f"Latency P50 exceeded: {results['latency_p50']:.1f} s"
    assert results["routing_accuracy"] >= 0.90, f"Routing accuracy too low: {results['routing_accuracy']:.2f}"
    assert results["log_integrity"] == 1.0, f"Log corruption detected"
    assert results["structured_output"] == 1.0, f"Pydantic validation failed"
    
    print("✅ All KPIs validated successfully")
    return results
```

---

## 🚀 Deployment Checklist

### Pre-Release

- [ ] All 79+ tests passing (`pytest tests/ -v`)
- [ ] KPI validation successful (`python tests/sarai_bench_v2_12.py`)
- [ ] Documentation complete (SKILL_DEVELOPMENT_GUIDE.md)
- [ ] CHANGELOG.md updated
- [ ] No regressions in v2.11 features

### Release

- [ ] Tag `v2.12.0` created
- [ ] GitHub Release published
- [ ] Docker image built (multi-arch)
- [ ] Cosign signature verified
- [ ] SBOM generated and published

### Post-Release

- [ ] Monitor first 100 skill executions
- [ ] Validate audit logs integrity
- [ ] Collect MCP feedback for threshold tuning
- [ ] Plan v2.13 (next skill additions)

---

## 📚 Documentation Deliverables

1. **SKILL_DEVELOPMENT_GUIDE.md** (~400 LOC)
   - How to create a new skill
   - Pydantic schema best practices
   - Testing patterns
   - Audit integration

2. **README_v2.12.md** (~200 LOC)
   - Updated architecture diagrams
   - Skill showcase examples
   - KPI comparison table

3. **CHANGELOG.md** (~100 LOC)
   - Breaking changes (none expected)
   - New features (6 skills)
   - Performance improvements

4. **Updated .github/copilot-instructions.md** (~150 LOC)
   - v2.12 patterns
   - Skill development conventions
   - Orchestrator usage

---

## 🎯 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Code Quality** | ≥ 90% coverage | pytest-cov |
| **Performance** | RAM ≤ 10.5 GB, Latency ≤ 22s | sarai_bench_v2_12.py |
| **Accuracy** | ≥ 90% routing | Manual validation + golden queries |
| **Security** | 100% log integrity | SHA-256 verification |
| **Modularity** | 6+ skills working | E2E tests |

---

## 🔮 Future Work (v2.13+)

### Potential New Skills

1. **Translation Skill**: Multi-language translation with quality metrics
2. **Math Skill**: Symbolic math solving with sympy integration
3. **Search Skill**: Web search orchestration (beyond RAG)
4. **Summarization Skill**: Multi-document summarization
5. **Vision Skill**: Image analysis (when Qwen-Omni multimodal ready)

### Technical Debt

- [ ] MCP auto-tuning for skill thresholds (currently static 0.4)
- [ ] Skill composition (chain multiple skills)
- [ ] Parallel skill execution (when RAM allows)
- [ ] Skill marketplace (community contributions)

---

## 📝 Notes

### Design Decisions

1. **¿Por qué Pydantic?**
   - Type-safe output garantizado
   - `model_json_schema()` evita escaping manual
   - Validación automática de campos
   - Integración nativa con FastAPI/LangChain

2. **¿Por qué Multi-label (no softmax)?**
   - Una query puede necesitar múltiples skills
   - Ejemplo: "Debug este código y explica por qué falla"
     → `programming: 0.85, diagnosis: 0.75`
   - Permite skill composition en v2.13+

3. **¿Por qué Empathy Layer separada?**
   - Skills producen output técnico estructurado
   - Omni-3B convierte a voz natural y empática
   - Permite reutilizar structured output en APIs
   - Modularidad: cambiar empathy sin tocar skills

4. **¿Por qué SHA-256 por skill?**
   - Auditoría granular (no solo sistema completo)
   - Detectar corrupción específica de skill
   - Permite validación incremental
   - Compliance con regulaciones (GDPR, SOC2)

---

## 🔗 References

- **Pydantic Docs**: https://docs.pydantic.dev/latest/
- **LangChain Structured Output**: https://python.langchain.com/docs/modules/model_io/output_parsers/
- **SARAi v2.11 Architecture**: `docs/LANGGRAPH_ARCHITECTURE.md`
- **TRM Paper**: https://github.com/SamsungSAILMontreal/TinyRecursiveModels

---

**End of Implementation Plan v2.12**

_Ready for technical review and implementation kickoff._
