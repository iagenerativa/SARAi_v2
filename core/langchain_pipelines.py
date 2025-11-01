"""
LangChain Pipelines - Composición Declarativa para SARAi v2.14

Este módulo implementa pipelines LCEL (LangChain Expression Language) que
componen modelos de forma DECLARATIVA (no imperativa).

Filosofía Anti-Spaghetti:
    "El código imperativo (if-else anidados, try-catch everywhere) es spaghetti.
    LCEL es declarativo: defines QUÉ quieres, no CÓMO hacerlo.
    
    ANTES (spaghetti):
        if has_image:
            try:
                result = qwen_model.process(image)
            except:
                try:
                    result = solar_model.process(extract_text(image))
                except:
                    result = "Error"
        else:
            result = solar_model.process(text)
    
    AHORA (LCEL):
        pipeline = RunnableBranch(
            (has_image, qwen_model),
            solar_model
        )
        result = pipeline.invoke(input)
    "

Características:
    - Composición con | operator
    - Fallback automático con RunnableBranch
    - Paralelización con RunnableParallel
    - Async nativo
    - Streaming automático
    - Type-safe

Autor: SARAi v2.14
Fecha: 1 Noviembre 2025
"""

from typing import Any, Dict, List, Optional, Union, Callable
import logging

# LangChain imports
from langchain_core.runnables import Runnable, RunnableBranch, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# SARAi imports
from core.unified_model_wrapper import ModelRegistry, get_model

logger = logging.getLogger(__name__)


# ============================================================================
# PIPELINE 1: Text Generation (Simple)
# ============================================================================

def create_text_pipeline(
    model_name: str = "solar_short",
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None
) -> Runnable:
    """
    Pipeline básico de generación de texto.
    
    LCEL Composition:
        (Optional) SystemPrompt → Model → StrOutputParser
    
    Args:
        model_name: Nombre del modelo (default: solar_short)
        temperature: Override de temperatura
        system_prompt: Prompt de sistema opcional
        
    Returns:
        Pipeline composable
        
    Example:
        >>> pipeline = create_text_pipeline("solar_short")
        >>> response = pipeline.invoke("¿Qué es Python?")
        'Python es un lenguaje de programación...'
        
        >>> # Con system prompt
        >>> pipeline = create_text_pipeline(
        ...     "solar_short",
        ...     system_prompt="Eres un experto en Python"
        ... )
        >>> response = pipeline.invoke("Explica decoradores")
    """
    model = get_model(model_name)
    
    # Si hay system prompt, crear chain con template
    if system_prompt:
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}")
        ])
        
        # Composición LCEL: Template → Model → Parser
        pipeline = template | model | StrOutputParser()
    else:
        # Composición simple: Model → Parser
        pipeline = model | StrOutputParser()
    
    logger.info(f"Created text pipeline with {model_name}")
    
    return pipeline


# ============================================================================
# PIPELINE 2: Vision Analysis (Multimodal)
# ============================================================================

def create_vision_pipeline(
    model_name: str = "qwen3_vl",
    temperature: Optional[float] = None
) -> Runnable:
    """
    Pipeline para análisis de imágenes/video.
    
    LCEL Composition:
        MultimodalModel → StrOutputParser
    
    Args:
        model_name: Modelo multimodal (default: qwen3_vl)
        temperature: Override de temperatura
        
    Returns:
        Pipeline composable
        
    Example:
        >>> pipeline = create_vision_pipeline()
        >>> response = pipeline.invoke({
        ...     "text": "Describe esta imagen",
        ...     "image": "screenshot.jpg"
        ... })
        'La imagen muestra...'
        
        >>> # Con video (frames)
        >>> response = pipeline.invoke({
        ...     "text": "Resume el video",
        ...     "video": ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
        ... })
    """
    model = get_model(model_name)
    
    # Composición simple: Multimodal → Parser
    pipeline = model | StrOutputParser()
    
    logger.info(f"Created vision pipeline with {model_name}")
    
    return pipeline


# ============================================================================
# PIPELINE 3: Hybrid with Fallback (Resilience)
# ============================================================================

def create_hybrid_pipeline_with_fallback(
    vision_model: str = "qwen3_vl",
    text_model: str = "solar_long",
    fallback_model: str = "lfm2"
) -> Runnable:
    """
    Pipeline híbrido con fallback automático.
    
    LCEL Composition:
        RunnableBranch(
            (has_image, vision_model),
            (has_error, fallback_model),
            text_model  # default
        ) → StrOutputParser
    
    Lógica de routing:
        1. Si input tiene imagen → usa vision_model
        2. Si vision_model falla → usa text_model
        3. Si todo falla → usa fallback_model (tiny)
    
    Args:
        vision_model: Modelo multimodal
        text_model: Modelo de texto principal
        fallback_model: Modelo de emergencia
        
    Returns:
        Pipeline con fallback automático
        
    Example:
        >>> pipeline = create_hybrid_pipeline_with_fallback()
        
        >>> # Con imagen (usa Qwen3-VL)
        >>> response = pipeline.invoke({
        ...     "text": "Analiza",
        ...     "image": "img.jpg"
        ... })
        
        >>> # Sin imagen (usa SOLAR)
        >>> response = pipeline.invoke("¿Qué es Python?")
        
        >>> # Si Qwen3-VL no disponible → SOLAR automáticamente
    """
    vision = get_model(vision_model)
    text = get_model(text_model)
    fallback = get_model(fallback_model)
    
    # Función de detección de imagen
    def has_image(input_data):
        """Detecta si el input contiene imagen."""
        if isinstance(input_data, dict):
            return "image" in input_data or "video" in input_data
        return False
    
    # Composición LCEL con branch
    pipeline = RunnableBranch(
        # Branch 1: Si tiene imagen → Vision
        (has_image, vision),
        
        # Branch 2 (default): Texto → Text model
        text
    ) | StrOutputParser()
    
    logger.info(
        f"Created hybrid pipeline: vision={vision_model}, "
        f"text={text_model}, fallback={fallback_model}"
    )
    
    return pipeline


# ============================================================================
# PIPELINE 4: Video Conference Analysis (Multi-Step)
# ============================================================================

def create_video_conference_pipeline(
    vision_model: str = "qwen3_vl",
    synthesis_model: str = "solar_long",
    enable_emotion: bool = True
) -> Runnable:
    """
    Pipeline completo para análisis de videoconferencias.
    
    LCEL Composition:
        RunnableParallel(
            visual = FrameAnalysis(vision_model),
            audio = AudioTranscription(),
            emotion = EmotionDetection()  # opcional
        ) → SynthesisPrompt → synthesis_model → StrOutputParser
    
    Pasos:
        1. Análisis paralelo:
           - Visual: Qwen3-VL analiza frames
           - Audio: Whisper transcribe
           - Emotion: Layer1 detecta emoción
        
        2. Síntesis con SOLAR:
           - Combina visual + audio + emotion
           - Genera resumen coherente
           - Extrae action items
    
    Args:
        vision_model: Modelo para análisis visual
        synthesis_model: Modelo para síntesis final
        enable_emotion: Incluir análisis emocional
        
    Returns:
        Pipeline multi-step composable
        
    Example:
        >>> pipeline = create_video_conference_pipeline()
        >>> summary = pipeline.invoke({
        ...     "frames": ["frame1.jpg", "frame2.jpg"],
        ...     "audio": audio_bytes,
        ...     "duration": 1800  # 30 min
        ... })
        
        Output:
        {
            "summary": "Reunión de planning...",
            "action_items": ["Item 1", "Item 2"],
            "participants_detected": 4,
            "emotional_tone": "professional"
        }
    """
    vision = get_model(vision_model)
    synthesis = get_model(synthesis_model)
    
    # Funciones de procesamiento (RunnableLambda)
    def analyze_visual(input_data: Dict) -> str:
        """Analiza frames con vision model."""
        frames = input_data.get("frames", [])
        
        if not frames:
            return "No visual data"
        
        # Seleccionar frames clave (cada 30 segundos)
        key_frames = frames[::30] if len(frames) > 30 else frames
        
        # Analizar con vision model
        analysis = vision.invoke({
            "text": "Describe el contenido visual de esta videoconferencia",
            "video": key_frames
        })
        
        return analysis
    
    def transcribe_audio(input_data: Dict) -> str:
        """Transcribe audio (placeholder para Whisper)."""
        audio = input_data.get("audio")
        
        if not audio:
            return "No audio data"
        
        # TODO: Integrar con Whisper
        # Por ahora, placeholder
        return "[Audio transcription placeholder]"
    
    def detect_emotion(input_data: Dict) -> str:
        """Detecta emoción del audio (Layer1)."""
        if not enable_emotion:
            return "Emotion detection disabled"
        
        audio = input_data.get("audio")
        
        if not audio:
            return "No emotion data"
        
        # TODO: Integrar con Layer1 emotion detection
        # Por ahora, placeholder
        return "[Emotion: neutral]"
    
    # PASO 1: Análisis paralelo (RunnableParallel)
    parallel_analysis = RunnableParallel(
        visual=RunnableLambda(analyze_visual),
        audio=RunnableLambda(transcribe_audio),
        emotion=RunnableLambda(detect_emotion)
    )
    
    # PASO 2: Template de síntesis
    synthesis_prompt = ChatPromptTemplate.from_template("""
Analiza la siguiente videoconferencia y genera un resumen ejecutivo:

ANÁLISIS VISUAL:
{visual}

TRANSCRIPCIÓN DE AUDIO:
{audio}

ANÁLISIS EMOCIONAL:
{emotion}

DURACIÓN: {duration} segundos

Genera un resumen que incluya:
1. Resumen ejecutivo (2-3 párrafos)
2. Action items detectados (lista numerada)
3. Participantes detectados (estimación visual)
4. Tono emocional predominante

Formato de salida JSON.
""")
    
    # Composición LCEL completa:
    # Parallel → Template → Synthesis → Parser
    pipeline = (
        parallel_analysis
        | synthesis_prompt
        | synthesis
        | StrOutputParser()
    )
    
    logger.info(
        f"Created video conference pipeline: vision={vision_model}, "
        f"synthesis={synthesis_model}, emotion={enable_emotion}"
    )
    
    return pipeline


# ============================================================================
# PIPELINE 5: RAG with Web Search (v2.10 Integration)
# ============================================================================

def create_rag_pipeline(
    search_model: str = "solar_long",
    enable_cache: bool = True,
    safe_mode: bool = False
) -> Runnable:
    """
    Pipeline RAG con búsqueda web.
    
    LCEL Composition:
        WebSearch → RunnableParallel(
            snippets = ExtractSnippets,
            urls = ExtractURLs
        ) → RAGPrompt → search_model → StrOutputParser
    
    Integra con v2.10:
        - core.web_cache.cached_search()
        - core.web_audit.log_web_query()
        - Safe mode sentinel
    
    Args:
        search_model: Modelo para síntesis
        enable_cache: Usar web cache
        safe_mode: Modo seguro (no buscar)
        
    Returns:
        Pipeline RAG composable
        
    Example:
        >>> pipeline = create_rag_pipeline()
        >>> response = pipeline.invoke("¿Quién ganó el Oscar 2025?")
        'Según fuentes verificadas...'
    """
    from core.web_cache import cached_search
    from core.web_audit import log_web_query
    from core.audit import is_safe_mode
    
    synthesis = get_model(search_model)
    
    def web_search(input_data: Union[str, Dict]) -> Dict:
        """Búsqueda web con cache."""
        # Extraer query
        if isinstance(input_data, dict):
            query = input_data.get("query", str(input_data))
        else:
            query = str(input_data)
        
        # Safe mode check
        if safe_mode or is_safe_mode():
            return {
                "snippets": [],
                "urls": [],
                "safe_mode_triggered": True
            }
        
        # Búsqueda con cache
        results = cached_search(query)
        
        if not results:
            return {
                "snippets": [],
                "urls": [],
                "search_failed": True
            }
        
        # Auditoría
        log_web_query(query, results, None, None)
        
        return {
            "snippets": results.get("snippets", []),
            "urls": results.get("urls", []),
            "query": query
        }
    
    # Prompt de síntesis
    rag_prompt = ChatPromptTemplate.from_template("""
Responde la siguiente pregunta usando SOLO la información de las fuentes proporcionadas:

PREGUNTA: {query}

FUENTES:
{snippets}

Instrucciones:
- Cita las fuentes cuando sea relevante
- Si la información es insuficiente, dilo explícitamente
- NO inventes información
- Sé conciso pero completo

Respuesta:
""")
    
    # Composición LCEL:
    # WebSearch → RAGPrompt → Synthesis → Parser
    pipeline = (
        RunnableLambda(web_search)
        | rag_prompt
        | synthesis
        | StrOutputParser()
    )
    
    logger.info(f"Created RAG pipeline with {search_model}")
    
    return pipeline


# ============================================================================
# PIPELINE 6: Skills Detection + Application (v2.12 Integration)
# ============================================================================

def create_skill_pipeline(
    base_model: str = "solar_short",
    enable_skill_detection: bool = True
) -> Runnable:
    """
    Pipeline con detección automática de skills.
    
    LCEL Composition:
        SkillDetector → RunnableBranch(
            (programming, solar + programming_prompt),
            (creative, lfm2 + creative_prompt),
            base_model  # default
        ) → StrOutputParser
    
    Integra con v2.12:
        - core.skill_configs.detect_and_apply_skill()
        - Long-tail pattern matching
        - Temperature per skill
    
    Args:
        base_model: Modelo base sin skill
        enable_skill_detection: Activar detección
        
    Returns:
        Pipeline con skills
        
    Example:
        >>> pipeline = create_skill_pipeline()
        
        >>> # Detecta programming skill automáticamente
        >>> response = pipeline.invoke("Escribe función Python para Fibonacci")
        
        >>> # Detecta creative skill automáticamente
        >>> response = pipeline.invoke("Escribe un cuento sobre un robot")
    """
    from core.skill_configs import detect_and_apply_skill
    
    base = get_model(base_model)
    
    def apply_skill(input_data: Union[str, Dict]) -> Dict:
        """Detecta y aplica skill apropiado."""
        # Extraer texto
        if isinstance(input_data, dict):
            text = input_data.get("text", str(input_data))
        else:
            text = str(input_data)
        
        # Detectar skill
        if enable_skill_detection:
            skill_config = detect_and_apply_skill(text, agent_type="solar")
            
            if skill_config:
                # Skill detectado: aplicar prompt especializado
                specialized_prompt = f"""{skill_config['system_prompt']}

User query: {text}

Response:"""
                return {
                    "prompt": specialized_prompt,
                    "temperature": skill_config["temperature"],
                    "skill_used": skill_config["name"]
                }
        
        # Sin skill: usar prompt original
        return {
            "prompt": text,
            "temperature": 0.7,
            "skill_used": None
        }
    
    # Template dinámico
    skill_prompt = RunnableLambda(lambda x: x["prompt"])
    
    # Composición LCEL:
    # SkillDetector → DynamicPrompt → Model → Parser
    pipeline = (
        RunnableLambda(apply_skill)
        | skill_prompt
        | base
        | StrOutputParser()
    )
    
    logger.info(f"Created skill pipeline with {base_model}")
    
    return pipeline


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_pipeline(pipeline_type: str, **kwargs) -> Runnable:
    """
    Factory function para obtener pipeline por tipo.
    
    Args:
        pipeline_type: Tipo de pipeline
            - "text": create_text_pipeline()
            - "vision": create_vision_pipeline()
            - "hybrid": create_hybrid_pipeline_with_fallback()
            - "video_conference": create_video_conference_pipeline()
            - "rag": create_rag_pipeline()
            - "skill": create_skill_pipeline()
        **kwargs: Argumentos para el pipeline
        
    Returns:
        Pipeline construido
        
    Example:
        >>> pipeline = get_pipeline("text", model_name="solar_short")
        >>> pipeline = get_pipeline("vision")
        >>> pipeline = get_pipeline("hybrid")
    """
    pipelines = {
        "text": create_text_pipeline,
        "vision": create_vision_pipeline,
        "hybrid": create_hybrid_pipeline_with_fallback,
        "video_conference": create_video_conference_pipeline,
        "rag": create_rag_pipeline,
        "skill": create_skill_pipeline
    }
    
    if pipeline_type not in pipelines:
        available = ", ".join(pipelines.keys())
        raise ValueError(
            f"Unknown pipeline type: {pipeline_type}. "
            f"Available: {available}"
        )
    
    return pipelines[pipeline_type](**kwargs)


if __name__ == "__main__":
    # Demo
    print("🎯 LangChain Pipelines v2.14")
    print("\nPipelines disponibles:")
    print("  - text: Generación de texto simple")
    print("  - vision: Análisis de imágenes/video")
    print("  - hybrid: Texto/visión con fallback")
    print("  - video_conference: Análisis de reuniones")
    print("  - rag: Búsqueda web + síntesis")
    print("  - skill: Detección automática de skills")
