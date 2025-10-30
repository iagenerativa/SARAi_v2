# SARAi Roadmap - v2.12 "Omni-Sentinel MoE" and Beyond

## 🎯 Current State (v2.11 "Omni-Sentinel")

**Released**: October 28, 2025  
**Status**: ✅ **Production Ready**

### Key Achievements

- ✅ **Voice-LLM Integration**: qwen25_7b_audio.onnx con latencia <250ms
- ✅ **Multi-language Support**: Español (nativo), Inglés (nativo), +6 idiomas vía NLLB
- ✅ **Audio Router**: Detección automática de idioma + fallback chain
- ✅ **Hardening**: Docker cap_drop ALL, read-only filesystem, no-new-privileges
- ✅ **Audit Trail**: HMAC SHA-256 para voz + web + sistema
- ✅ **KPIs**: RAM P99 10.8GB, Latency P50 19.5s, MOS 4.38, Disponibilidad 100%

**Total**: 77 tests, 2,584 LOC, Score Hardening 99/100

---

## 🚀 v2.12 "Omni-Sentinel MoE" (Next Release)

**Target Date**: November 21, 2025 (14 días)  
**Status**: 🔴 **PLANNED**

### Vision

> _"SARAi no sólo responde: especializa, valida, audita y empatiza, convirtiendo cada skill en un módulo trazable, seguro y reutilizable."_

### Core Features

#### 1. Mixture of Experts (MoE) Architecture
- **6 Specialized Skills**:
  1. **Programming**: Code generation, debugging, testing, complexity analysis
  2. **Diagnosis**: Problem troubleshooting, root cause analysis
  3. **Finance**: Budget planning, ROI calculations, investment analysis
  4. **Logic**: Logical reasoning, if-then inference, proof validation
  5. **Creative**: Story/poem generation, creative design
  6. **Reasoning**: Strategic analysis, decision-making, prioritization

- **Structured Output**: Pydantic schemas per skill (type-safe, validated)
- **TRM-Router Multi-Skill**: 8-way classification (hard, soft, 6 skills)
- **Empathy Layer**: Omni-3B converts structured → natural voice

#### 2. Enhanced Audit Trail
- **SHA-256 per Skill**: Granular corruption detection
- **Skill-specific Logs**: `logs/skills/{skill_name}_{date}.jsonl`
- **100% Integrity**: All skill executions verified

#### 3. Advanced Orchestration
- **Skill Selection**: MCP-learned thresholds (default 0.4)
- **Model Selection**: expert_short (hard > 0.7) vs tiny (soft)
- **Fallback Chain**: Skill → v2.11 routing → sentinel

### Technical Goals

| Component | Target | Benefit |
|-----------|--------|---------|
| **Skills** | 6+ modular | Domain specialization |
| **Routing Accuracy** | ≥ 90% | Better intent detection |
| **Structured Output** | 100% Pydantic | Type-safe responses |
| **RAM P99** | ≤ 10.5 GB | No degradation vs v2.11 |
| **Latency P50** | ≤ 22 s | Acceptable for expert_short |
| **Audit Granularity** | Per-skill | Regulatory compliance |

### Implementation Phases

```
M1: Base Skill Architecture          (Day 1-2)   → 300 LOC, 5 tests
M2: TRM-Router Multi-Skill           (Day 3-4)   → 500 LOC, 8 tests
M3: Skills Implementation            (Day 5-8)   → 1600 LOC, 25 tests
M4: Orchestrator Integration         (Day 9-10)  → 550 LOC, 10 tests
M5: Audit & Security                 (Day 11)    → 320 LOC, 6 tests
M6: E2E Testing & Validation         (Day 12-13) → 400 LOC, 20 tests
M7: Release & Documentation          (Day 14)    → 850 LOC docs
```

**Total**: 79+ tests, 4,520 LOC (3,670 producción + 850 tests)

### Breaking Changes
**None expected** - Full backward compatibility with v2.11

---

## 🔮 v2.13 "Skill Composer" (Future - Q1 2026)

**Target**: January 2026  
**Focus**: Multi-skill orchestration and skill marketplace

### Planned Features

#### 1. Skill Composition
- **Chain Multiple Skills**: `programming → diagnosis → creative`
- **Parallel Execution**: Run multiple skills concurrently (if RAM allows)
- **Skill Dependencies**: Declarative dependency graphs

Example:
```python
# "Debug este código y escribe una historia sobre el bug"
workflow = SkillComposition([
    ("programming", {"mode": "debug"}),
    ("diagnosis", {"input": "programming.errors"}),
    ("creative", {"input": "diagnosis.root_cause", "style": "humorous"})
])
```

#### 2. MCP Auto-Tuning
- **Dynamic Thresholds**: Learn optimal thresholds per skill from feedback
- **A/B Testing**: Experiment with different routing strategies
- **Performance Metrics**: Track skill accuracy, latency, user satisfaction

#### 3. Skill Marketplace
- **Community Skills**: Users contribute custom skills
- **Skill Registry**: Central repository with versioning
- **Validation Pipeline**: Automated testing before publication

#### 4. Additional Skills
- **Translation**: Multi-language with quality metrics (beyond NLLB)
- **Math**: Symbolic math solving (sympy integration)
- **Search**: Advanced web search orchestration
- **Summarization**: Multi-document summarization
- **Vision**: Image analysis (when Qwen-Omni multimodal ready)

### Technical Goals

| Feature | Target | Benefit |
|---------|--------|---------|
| **Skill Composition** | 3+ chained skills | Complex workflows |
| **Parallel Skills** | 2 concurrent (if RAM < 10GB) | Lower latency |
| **MCP Auto-Tune** | Thresholds learned in 100 samples | Better accuracy |
| **Community Skills** | 10+ published | Ecosystem growth |
| **New Skills** | +5 (total 11+) | Broader capabilities |

---

## 🌟 v2.14 "Federated MoE" (Future - Q2 2026)

**Target**: April 2026  
**Focus**: Distributed skill execution and privacy-first architecture

### Planned Features

#### 1. Federated Skill Execution
- **Remote Skills**: Execute skills on external servers (opt-in)
- **Privacy Guarantees**: Data encryption, no logging on remote
- **Latency Optimization**: CDN-like skill distribution

#### 2. Skill Caching
- **Result Cache**: Cache common skill outputs (e.g., "What is Python?")
- **Embedding-based Lookup**: Semantic similarity for cache hits
- **TTL per Skill**: Different cache lifetimes per skill type

#### 3. Edge Deployment
- **Raspberry Pi 5**: Full SARAi on 8GB RAM
- **ONNX Quantization**: All models → ONNX Q4 (RAM -30%)
- **Skill Pruning**: Only load needed skills on edge

#### 4. Enterprise Features
- **Multi-Tenancy**: Isolated skill execution per tenant
- **Role-Based Skills**: Different skills for different user roles
- **Compliance**: GDPR, HIPAA, SOC2 ready

### Technical Goals

| Feature | Target | Benefit |
|---------|--------|---------|
| **Federated Skills** | 3+ remote providers | Scalability |
| **Cache Hit Rate** | 40-60% | Lower latency |
| **Edge Deployment** | Pi-5 8GB | Broader hardware support |
| **Multi-Tenancy** | 100+ tenants | Enterprise adoption |
| **RAM Optimization** | 6GB P99 (via ONNX Q4) | Better resource usage |

---

## 📊 Long-Term Roadmap (2026+)

### Q3 2026: v2.15 "Contextual Memory"
- **Persistent Context**: Long-term memory across sessions
- **Personalization**: User-specific skill preferences
- **Conversation Graphs**: Semantic graphs of user interactions

### Q4 2026: v2.16 "Multimodal MoE"
- **Vision Skills**: Image analysis, OCR, object detection
- **Audio Skills**: Music generation, voice cloning
- **Video Skills**: Video summarization, scene detection

### 2027: v3.0 "AGI Local"
- **Self-Improving**: Auto-generate training data
- **Meta-Learning**: Learn new skills from few examples
- **Emergent Behaviors**: Skills discover new capabilities

---

## 🎯 Success Metrics (Continuous)

### Performance KPIs

| Metric | v2.11 | v2.12 Target | v2.13 Target | v2.14 Target |
|--------|-------|--------------|--------------|--------------|
| **RAM P99** | 10.8 GB | ≤ 10.5 GB | ≤ 10.0 GB | ≤ 6.0 GB |
| **Latency P50** | 19.5 s | ≤ 22 s | ≤ 18 s | ≤ 15 s |
| **Test Coverage** | 77 tests | 79+ tests | 100+ tests | 150+ tests |
| **Skills** | 0 | 6 | 11+ | 15+ |
| **Routing Accuracy** | 87% | 90% | 92% | 95% |

### Adoption Metrics

| Metric | 2025 Target | 2026 Target | 2027 Target |
|--------|-------------|-------------|-------------|
| **GitHub Stars** | 100 | 500 | 1000 |
| **Contributors** | 5 | 20 | 50 |
| **Community Skills** | 0 | 10 | 50 |
| **Enterprise Deployments** | 0 | 5 | 20 |
| **Edge Devices** | 0 | 100 | 1000 |

---

## 🛠️ Technical Debt & Improvements

### High Priority (v2.12)
- [ ] MCP threshold learning (currently hardcoded 0.4)
- [ ] Skill output validation (robust against malformed JSON)
- [ ] RAM profiling per skill (ensure +0.1-0.3 GB target)

### Medium Priority (v2.13)
- [ ] Skill composition graph executor
- [ ] Parallel skill execution (requires RAM headroom detection)
- [ ] A/B testing framework for routing

### Low Priority (v2.14)
- [ ] ONNX Q4 conversion for all models
- [ ] Federated skill protocol design
- [ ] Multi-tenancy isolation testing

---

## 🔗 Dependencies & Prerequisites

### v2.12 Prerequisites
- ✅ v2.11 stable and released
- ✅ Pydantic ≥ 2.0 (for `model_json_schema()`)
- ✅ LangChain ≥ 0.1.0 (for orchestration)
- ✅ SOLAR + LFM2 GGUF models downloaded

### v2.13 Prerequisites
- ⏸️ v2.12 released with 90%+ routing accuracy
- ⏸️ MCP learning pipeline validated
- ⏸️ Community skill submission process defined

### v2.14 Prerequisites
- ⏸️ v2.13 skill composition stable
- ⏸️ ONNX Q4 conversion scripts ready
- ⏸️ Federated skill protocol designed

---

## 📝 Design Principles (Immutable)

All future versions MUST adhere to:

1. **CPU-First**: Zero GPU dependency, optimized for consumer hardware
2. **RAM Budget**: Never exceed 12GB RAM (16GB system - 4GB OS)
3. **Security-by-Design**: All features auditable, zero-trust by default
4. **Modular**: Each skill/component is independently testable
5. **Empathy-First**: Technical accuracy + emotional intelligence
6. **Open Source**: Apache 2.0, community-driven development

---

## 🎉 Community Roadmap

### How to Contribute

1. **Submit Skills**: Fork → Add skill → PR with tests
2. **Report Issues**: GitHub Issues with reproducible examples
3. **Suggest Features**: Discussions tab with use cases
4. **Documentation**: Improve guides, add examples
5. **Testing**: Beta test pre-releases, report bugs

### Recognition

- **Top Contributors**: Featured in README
- **Skill Authors**: Credit in skill metadata
- **Bug Bounty**: Critical bugs eligible for rewards

---

## 📅 Release Calendar

| Version | Target Date | Status | Focus |
|---------|-------------|--------|-------|
| v2.11 | ✅ Oct 28, 2025 | Released | Omni-Sentinel (Voice) |
| v2.12 | 🎯 Nov 21, 2025 | Planned | Omni-Sentinel MoE (Skills) |
| v2.13 | 📅 Jan 2026 | Future | Skill Composer |
| v2.14 | 📅 Apr 2026 | Future | Federated MoE |
| v2.15 | 📅 Jul 2026 | Future | Contextual Memory |
| v2.16 | 📅 Oct 2026 | Future | Multimodal MoE |
| v3.0 | 📅 2027 | Future | AGI Local |

---

**Last Updated**: November 7, 2025  
**Maintained By**: SARAi Core Team  
**License**: Apache 2.0

_This roadmap is a living document. Dates and features subject to change based on community feedback and technical feasibility._
