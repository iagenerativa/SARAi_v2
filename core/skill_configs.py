"""
Skill Configurations - Prompts especializados para SOLAR/LFM2
=============================================================
NO carga modelos adicionales. Cada skill es un conjunto de:
- System prompts optimizados
- Parámetros de generación (temperature, max_tokens, top_p)
- Keywords para routing automático del TRM
- Dominios de expertise
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class SkillConfig:
    """Configuración de un skill (NO un modelo, solo prompts + params)"""
    
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        keywords: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        preferred_model: str = "solar",  # "solar" o "lfm2"
        use_case: str = "general"
    ):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.keywords = keywords
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.preferred_model = preferred_model
        self.use_case = use_case
    
    def build_prompt(self, user_query: str) -> str:
        """Construye el prompt completo con system + user query"""
        return f"{self.system_prompt}\n\nUser: {user_query}\nAssistant:"
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Retorna parámetros de generación optimizados para este skill"""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": ["User:", "\n\n"]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa config para persistencia"""
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "keywords": self.keywords,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "preferred_model": self.preferred_model,
            "use_case": self.use_case
        }


# ========================================
# SKILL DEFINITIONS
# ========================================

PROGRAMMING_SKILL = SkillConfig(
    name="programming",
    description="Experto en programación y desarrollo de software (VisCoder2-7B)",
    system_prompt="""You are VisCoder2, an expert programming assistant with deep knowledge of:
- Multiple programming languages (Python, JavaScript, Rust, Go, C++, Java)
- Software design patterns and best practices
- Algorithm optimization and complexity analysis
- Debugging and code review
- Testing strategies (unit, integration, E2E)

Your responses should be:
- Technically precise and well-documented
- Include code examples with clear explanations
- Consider edge cases and error handling
- Follow language-specific best practices
- Provide performance considerations when relevant
- Include type hints (Python) or JSDoc (JavaScript) where applicable""",
    keywords=["code", "programming", "función", "clase", "debug", "test", "algoritmo", "código", "implementa"],
    temperature=0.3,  # Baja para código preciso
    max_tokens=3072,
    top_p=0.85,
    preferred_model="viscoder2",  # ✨ Usa VisCoder2 especializado
    use_case="code_generation"
)

DIAGNOSIS_SKILL = SkillConfig(
    name="diagnosis",
    description="Experto en diagnóstico y análisis de problemas",
    system_prompt="""You are a diagnostic expert specialized in:
- System troubleshooting and root cause analysis
- Problem identification and classification
- Step-by-step diagnostic procedures
- Evidence-based reasoning
- Solution recommendation

Your diagnostic approach:
1. Gather symptoms and observations
2. Formulate hypotheses
3. Test hypotheses systematically
4. Identify root cause
5. Recommend solutions with priority levels

Be methodical, precise, and always provide reasoning for your conclusions.""",
    keywords=["problema", "error", "diagnóstico", "fallo", "bug", "soluciona", "analiza", "debugging"],
    temperature=0.4,
    max_tokens=2560,
    top_p=0.9,
    preferred_model="solar",
    use_case="diagnostic"
)

FINANCIAL_SKILL = SkillConfig(
    name="financial",
    description="Experto en análisis y gestión financiera",
    system_prompt="""You are a financial expert with expertise in:
- Financial analysis and metrics (ROI, NPV, IRR, ratios)
- Investment strategies and portfolio management
- Risk assessment and mitigation
- Budgeting and forecasting
- Financial reporting and compliance

Your analysis should:
- Use quantitative methods and data-driven insights
- Consider risk-reward tradeoffs
- Provide actionable recommendations
- Include relevant financial metrics
- Be clear about assumptions and limitations

Always present financial data in structured formats (tables, bullet points).""",
    keywords=["financiero", "inversión", "presupuesto", "ROI", "análisis", "riesgo", "portfolio", "capital"],
    temperature=0.5,
    max_tokens=2048,
    top_p=0.88,
    preferred_model="solar",
    use_case="financial_analysis"
)

CREATIVE_SKILL = SkillConfig(
    name="creative",
    description="Experto en tareas creativas y generación de contenido",
    system_prompt="""You are a creative expert specialized in:
- Content generation (stories, articles, marketing copy)
- Ideation and brainstorming
- Creative problem solving
- Narrative development and storytelling
- Artistic direction and conceptualization

Your creative outputs should:
- Be original and engaging
- Explore multiple perspectives and variations
- Balance creativity with clarity
- Adapt tone and style to the context
- Provide reasoning for creative choices

Embrace divergent thinking while maintaining coherence.""",
    keywords=["crea", "genera", "escribe", "historia", "idea", "concepto", "diseña", "innova", "creativo"],
    temperature=0.9,  # Alta para creatividad
    max_tokens=3584,
    top_p=0.95,
    preferred_model="lfm2",  # LFM2 mejor para soft/creative
    use_case="creative_generation"
)

REASONING_SKILL = SkillConfig(
    name="reasoning",
    description="Experto en razonamiento avanzado y resolución de problemas complejos",
    system_prompt="""You are a reasoning expert specialized in:
- Chain of Thought (CoT) reasoning
- Tree of Thought exploration
- Metacognitive analysis
- Problem decomposition
- Strategic thinking and planning

Your reasoning approach:
1. Break down complex problems into components
2. Explore multiple solution paths
3. Evaluate alternatives systematically
4. Reflect on your own reasoning process
5. Provide clear step-by-step explanations

Always make your thinking process transparent and verifiable.""",
    keywords=["razonamiento", "pensamiento", "analiza", "estrategia", "complejo", "resuelve", "metodología"],
    temperature=0.6,
    max_tokens=2560,
    top_p=0.92,
    preferred_model="solar",
    use_case="complex_reasoning"
)

CTO_SKILL = SkillConfig(
    name="cto",
    description="Experto en visión tecnológica y dirección técnica",
    system_prompt="""You are a CTO-level technology expert with deep knowledge of:
- Technology strategy and roadmap planning
- System architecture and scalability
- Innovation and emerging technologies
- Technical leadership and team building
- DevOps, infrastructure, and cloud platforms

Your strategic guidance should:
- Balance innovation with pragmatism
- Consider long-term maintainability
- Evaluate technology tradeoffs
- Align technical decisions with business goals
- Provide implementation guidance

Think at both strategic (vision) and tactical (execution) levels.""",
    keywords=["arquitectura", "infraestructura", "tecnología", "roadmap", "escalabilidad", "cloud", "devops"],
    temperature=0.5,
    max_tokens=2048,
    top_p=0.88,
    preferred_model="solar",
    use_case="technical_strategy"
)

SRE_SKILL = SkillConfig(
    name="sre",
    description="Experto en fiabilidad y operaciones de sistemas",
    system_prompt="""You are an SRE (Site Reliability Engineering) expert specialized in:
- System reliability and high availability
- Incident management and post-mortems
- Monitoring, observability, and alerting
- Capacity planning and performance optimization
- Automation and infrastructure as code

Your approach:
- Prioritize reliability and fault tolerance
- Use data-driven decision making
- Implement defense in depth
- Automate repetitive tasks
- Document runbooks and procedures

Focus on operational excellence and continuous improvement.""",
    keywords=["kubernetes", "k8s", "monitoring", "reliability", "sre", "deployment", "automation", "observability"],
    temperature=0.4,
    max_tokens=2048,
    top_p=0.87,
    preferred_model="solar",
    use_case="sre_operations"
)

# Registro de todos los skills
ALL_SKILLS = {
    "programming": PROGRAMMING_SKILL,
    "diagnosis": DIAGNOSIS_SKILL,
    "financial": FINANCIAL_SKILL,
    "creative": CREATIVE_SKILL,
    "reasoning": REASONING_SKILL,
    "cto": CTO_SKILL,
    "sre": SRE_SKILL
}


def get_skill(skill_name: str) -> Optional[SkillConfig]:
    """Obtiene configuración de un skill por nombre"""
    return ALL_SKILLS.get(skill_name.lower())


def list_skills() -> List[str]:
    """Lista todos los skills disponibles"""
    return list(ALL_SKILLS.keys())


def match_skill_by_keywords(query: str) -> Optional[SkillConfig]:
    """
    Encuentra el skill más apropiado basado en keywords en la query
    Usa long-tail matching con pesos para mayor precisión
    """
    query_lower = query.lower()
    
    # Long-tail patterns: combinaciones de palabras que son muy específicas
    # Peso alto = match casi garantizado
    longtail_patterns = {
        "programming": [
            ("código", "python", 3.0),
            ("función", "implementa", 3.0),
            ("algoritmo", "código", 3.0),
            ("debug", "error", 2.5),
            ("refactoriza", "código", 2.5),
        ],
        "diagnosis": [
            ("diagnostica", "problema", 3.0),
            ("analiza", "error", 3.0),
            ("fallo", "sistema", 2.5),
            ("memory leak", 3.0),  # Frase específica
            ("bug", "producción", 2.5),
        ],
        "financial": [
            ("análisis", "financiero", 3.0),
            ("roi", "inversión", 3.0),
            ("presupuesto", "costos", 2.5),
            ("margen", "beneficio", 2.5),
            ("flujo", "caja", 2.5),
        ],
        "creative": [
            ("crea", "historia", 3.0),
            ("escribe", "narrativa", 3.0),
            ("genera", "ideas", 2.5),
            ("brainstorm", 2.5),
            ("innovador", "concepto", 2.5),
        ],
        "reasoning": [
            ("razonamiento", "lógico", 3.0),
            ("analiza", "estrategia", 2.5),
            ("problema", "complejo", 2.5),
            ("paso", "paso", 2.0),
        ],
        "cto": [
            ("arquitectura", "sistema", 3.0),
            ("roadmap", "técnico", 3.0),
            ("escalabilidad", "infraestructura", 2.5),
            ("stack", "tecnológico", 2.5),
        ],
        "sre": [
            ("kubernetes", "cluster", 3.0),
            ("monitoring", "alertas", 3.0),
            ("reliability", "sla", 2.5),
            ("incident", "postmortem", 2.5),
        ],
    }
    
    # Contar coincidencias por skill con pesos
    scores = {}
    
    # 1. Verificar long-tail patterns primero (alta confianza)
    for skill_name, patterns in longtail_patterns.items():
        skill_score = 0.0
        
        for pattern in patterns:
            if len(pattern) == 2:
                # Patrón de frase única
                phrase, weight = pattern
                if phrase in query_lower:
                    skill_score += weight
            else:
                # Patrón de combinación
                word1, word2, weight = pattern
                if word1 in query_lower and word2 in query_lower:
                    skill_score += weight
        
        if skill_score > 0:
            scores[skill_name] = skill_score
    
    # 2. Si hay match long-tail con score alto, retornar inmediatamente
    if scores:
        best_score = max(scores.values())
        if best_score >= 2.5:  # Umbral de confianza alta
            best_skill_name = max(scores, key=scores.get)
            return ALL_SKILLS[best_skill_name]
    
    # 3. Fallback: keyword matching simple con pesos
    for skill_name, skill_config in ALL_SKILLS.items():
        if skill_name not in scores:
            scores[skill_name] = 0.0
        
        # Contar keywords simples (peso 1.0 cada una)
        for keyword in skill_config.keywords:
            if keyword in query_lower:
                scores[skill_name] += 1.0
    
    # 4. Filtrar skills sin matches
    scores = {k: v for k, v in scores.items() if v > 0}
    
    if not scores:
        return None
    
    # 5. Retornar skill con mayor score
    best_skill_name = max(scores, key=scores.get)
    return ALL_SKILLS[best_skill_name]


def get_skills_info() -> Dict[str, Dict[str, Any]]:
    """Retorna información resumida de todos los skills"""
    return {
        name: {
            "description": skill.description,
            "keywords": skill.keywords,
            "temperature": skill.temperature,
            "preferred_model": skill.preferred_model,
            "use_case": skill.use_case
        }
        for name, skill in ALL_SKILLS.items()
    }


__all__ = [
    "SkillConfig",
    "PROGRAMMING_SKILL",
    "DIAGNOSIS_SKILL", 
    "FINANCIAL_SKILL",
    "CREATIVE_SKILL",
    "REASONING_SKILL",
    "CTO_SKILL",
    "SRE_SKILL",
    "ALL_SKILLS",
    "get_skill",
    "list_skills",
    "match_skill_by_keywords",
    "get_skills_info"
]
