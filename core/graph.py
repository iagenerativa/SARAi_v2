"""
Orquestador LangGraph para SARAi v2
Flujo de estado: detect_input → [audio→process_voice] → classify → mcp → route → generate → [audio→tts] → feedback
"""

from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
import torch

from core.embeddings import get_embedding_model
from core.trm_classifier import create_trm_classifier, TRMClassifierSimulated
from core.mcp import create_mcp
from core.feedback import get_feedback_detector
from core.omni_loop import get_omni_loop  # NEW v2.16: Reflexive Loop
from agents.expert_agent import get_expert_agent
from agents.tiny_agent import get_tiny_agent
# DEPRECATED v2.16: multimodal_agent reemplazado por omni_native (GGUF más eficiente)
# from agents.multimodal_agent import get_multimodal_agent, MultimodalAgent
from agents.rag_agent import create_rag_node  # NEW v2.10
from agents.omni_native import get_omni_agent  # NEW v2.16: Omni-7B LangChain
from agents.qwen3_vl import get_qwen3_vl_agent  # NEW v2.16.1: Vision specialist


class State(TypedDict):
    """Estado compartido en el flujo LangGraph (v2.16.1 con Vision)"""
    # Input original
    input: str
    
    # Voice fields (v2.11)
    input_type: Literal["text", "audio", "image", "video"]  # v2.16.1: + image/video
    audio_input: Optional[bytes]
    audio_output: Optional[bytes]
    detected_emotion: Optional[str]  # "empático" | "neutral" | "urgente"
    detected_lang: Optional[str]     # ISO 639-1 code
    voice_metadata: dict
    
    # Vision fields (v2.16.1)
    image_path: Optional[str]
    video_path: Optional[str]
    fps: Optional[float]             # Para video analysis
    
    # Omni-Loop fields (v2.16)
    enable_reflection: bool          # Activar auto-corrección
    omni_loop_iterations: Optional[list]  # Historia de iteraciones
    auto_corrected: bool             # True si hubo corrección
    
    # Classification scores
    hard: float
    soft: float
    web_query: float  # v2.10
    
    # MCP weights
    alpha: float
    beta: float
    
    # Execution
    agent_used: Literal["expert", "tiny", "multimodal", "rag", "omni", "vision", "omni_loop"]  # v2.16: + omni_loop
    response: str
    feedback: float
    
    # Metadata
    rag_metadata: dict  # v2.10: metadata de búsqueda web


class SARAiOrchestrator:
    """
    Orquestador principal de SARAi
    Gestiona el flujo completo de procesamiento
    """
    
    def __init__(self, use_simulated_trm: bool = False):
        """
        Args:
            use_simulated_trm: Si True, usa TRM simulado (antes de entrenar el real)
        """
        print("🚀 Inicializando SARAi v2...")
        
        # TRM-Classifier (real o simulado)
        if use_simulated_trm:
            print("⚠️  Usando TRM-Classifier simulado")
            self.trm_classifier = TRMClassifierSimulated()
            self.embedding_model = None  # No necesario con TRM simulado
        else:
            self.embedding_model = get_embedding_model()
            self.trm_classifier = create_trm_classifier()
        
        self.mcp = create_mcp()
        self.feedback_detector = get_feedback_detector()
        
        # Agentes LLM (carga bajo demanda)
        self.expert_agent = get_expert_agent()
        self.tiny_agent = get_tiny_agent()
        # DEPRECATED v2.16: multimodal_agent eliminado (solapaba con omni_native)
        # self.multimodal_agent = get_multimodal_agent()
        
        # v2.16.1 Best-of-Breed: Omni-3B permanente (audio español + NLLB)
        # Solo para audio, empatía la maneja tiny (LFM2)
        print("🌟 Cargando Omni-3B Agent (audio permanente)...")
        self.omni_agent = get_omni_agent()
        
        # NEW v2.10: RAG Agent (requiere model_pool)
        from core.model_pool import get_model_pool
        self.model_pool = get_model_pool()
        self.rag_node = create_rag_node(self.model_pool)
        
        # NEW v2.16: Omni-Loop (reflexive engine)
        print("🔄 Cargando Omni-Loop Engine (reflexive)...")
        self.omni_loop = get_omni_loop()
        
        # Construir grafo
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()
        
        print("✅ SARAi listo")
    
    def _build_graph(self) -> StateGraph:
        """Construye el grafo de estado de LangGraph (v2.16.1 con Vision)"""
        workflow = StateGraph(State)
        
        # Nodos (v2.16.1: añadido generate_vision)
        workflow.add_node("detect_input_type", self._detect_input_type)
        workflow.add_node("process_voice", self._process_voice)
        workflow.add_node("classify", self._classify_intent)
        workflow.add_node("mcp", self._compute_weights)
        workflow.add_node("execute_omni_loop", self._execute_omni_loop)  # NEW v2.16: Reflexive loop
        workflow.add_node("generate_expert", self._generate_expert)
        workflow.add_node("generate_tiny", self._generate_tiny)
        workflow.add_node("generate_omni", self._generate_omni)  # NEW v2.16
        workflow.add_node("generate_vision", self._generate_vision)  # NEW v2.16.1
        workflow.add_node("execute_rag", self.rag_node)
        workflow.add_node("enhance_with_emotion", self._enhance_with_emotion)  # v2.11
        workflow.add_node("generate_tts", self._generate_tts)  # v2.11
        workflow.add_node("feedback", self._log_feedback)
        
        # Flujo (v2.11: entrada por detect_input_type)
        workflow.set_entry_point("detect_input_type")
        
        # Routing de input: audio vs texto vs vision
        workflow.add_conditional_edges(
            "detect_input_type",
            self._route_by_input_type,
            {
                "audio": "process_voice",
                "text": "classify",
                "vision": "generate_vision"  # NEW v2.16.1: visión directo (sin classify)
            }
        )
        
        # Audio procesado → classify
        workflow.add_edge("process_voice", "classify")
        
        # Clasificación → MCP
        workflow.add_edge("classify", "mcp")
        
        # Routing condicional basado en web_query (v2.10)
        workflow.add_conditional_edges(
            "mcp",
            self._route_to_agent,
            {
                "expert": "generate_expert",
                "tiny": "generate_tiny",
                "omni": "generate_omni",  # NEW v2.16
                "omni_loop": "execute_omni_loop",  # NEW v2.16: Reflexive multimodal
                "rag": "execute_rag"
            }
        )
        
        # LLMs → enhance_with_emotion
        workflow.add_edge("generate_expert", "enhance_with_emotion")
        workflow.add_edge("generate_tiny", "enhance_with_emotion")
        workflow.add_edge("generate_omni", "enhance_with_emotion")  # NEW v2.16
        workflow.add_edge("execute_omni_loop", "enhance_with_emotion")  # NEW v2.16: Reflexive loop
        workflow.add_edge("generate_vision", "enhance_with_emotion")  # NEW v2.16.1
        workflow.add_edge("execute_rag", "enhance_with_emotion")
        
        # Emotion → TTS (condicional si audio_input)
        workflow.add_conditional_edges(
            "enhance_with_emotion",
            self._route_to_tts,
            {
                "tts": "generate_tts",
                "skip": "feedback"
            }
        )
        
        # TTS → feedback
        workflow.add_edge("generate_tts", "feedback")
        workflow.add_edge("feedback", END)
        
        return workflow
    
    def _classify_intent(self, state: State) -> dict:
        """Nodo: Clasificar hard/soft/web_query intent (v2.10)"""
        user_input = state["input"]
        
        # Clasificar según tipo de TRM
        if isinstance(self.trm_classifier, TRMClassifierSimulated):
            # TRM simulado: usa keywords directamente
            scores = self.trm_classifier.invoke(user_input)
        else:
            # TRM real: requiere embeddings
            embedding = self.embedding_model.encode(user_input)
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
            scores = self.trm_classifier.invoke(embedding_tensor)
        
        print(f"📊 Intent: hard={scores['hard']:.2f}, soft={scores['soft']:.2f}, web_query={scores.get('web_query', 0.0):.2f}")
        
        return {
            "hard": scores["hard"],
            "soft": scores["soft"],
            "web_query": scores.get("web_query", 0.0)  # v2.10
        }
    
    def _compute_weights(self, state: State) -> dict:
        """Nodo: Calcular pesos α/β con MCP"""
        alpha, beta = self.mcp.compute_weights(state["hard"], state["soft"])
        
        print(f"⚖️  Pesos: α={alpha:.2f} (hard), β={beta:.2f} (soft)")
        
        return {"alpha": alpha, "beta": beta}
    
    def _route_to_agent(self, state: State) -> str:
        """
        Enrutamiento inteligente basado en scores y tipo de input.
        
        PRIORIDADES v2.16 con Omni-Loop:
        1. RAG si web_query > 0.7 (búsqueda web)
        2. Visión si image_path o video_path (Qwen3-VL-4B on-demand)
        3. Omni-Loop si imagen Y texto (multimodal reflexivo)
        4. Audio si input_type == "audio" (Omni-3B para español, NLLB para otros)
        5. Expert si alpha > 0.7 (SOLAR HTTP para razonamiento técnico)
        6. Tiny (LFM2) fallback (empathy soft > 0.7 + general)
        
        ARQUITECTURA:
        - Omni-Loop: skill_draft + skill_image (Phoenix) - 3 iteraciones reflexivas
        - Audio: Qwen3-VL-4B-Instruct (190 MB) - Solo español STT/TTS
        - Traducción: NLLB-600M - Pipeline multilingüe (STT→translate→ES→process→translate→TTS)
        - Empatía: LFM2-1.2B (700 MB) - Soft-skills, soft > 0.7
        - Expert: SOLAR-10.7B HTTP - Hard-skills, alpha > 0.7
        - Visión: Qwen3-VL-4B (3.3 GB) - Image/video on-demand
        
        Returns:
            Nombre del agente: "rag" | "vision" | "omni_loop" | "omni" | "expert" | "tiny"
        """
        # PRIORIDAD 1: RAG para búsqueda web
        if state.get("web_query", 0.0) > 0.7:
            print("🔍 Ruta: RAG Agent (búsqueda web)")
            return "rag"
        
        # PRIORIDAD 2: Visión para imagen/video sin texto significativo (Qwen3-VL-4B)
        # Si hay imagen PERO también texto → usar Omni-Loop (multimodal reflexivo)
        if state.get("image_path") or state.get("video_path"):
            # Detectar si hay texto significativo (>20 chars)
            text_length = len(state.get("input", "").strip())
            
            if text_length > 20:
                # Multimodal reflexivo: imagen + texto → Omni-Loop
                print("🔄 Ruta: Omni-Loop (multimodal reflexivo)")
                return "omni_loop"
            else:
                # Solo imagen/video → Vision specialist
                print("👁️ Ruta: Vision Agent (Qwen3-VL-4B)")
                return "vision"
        
        # PRIORIDAD 3: Audio (Omni-3B para español, NLLB para otros)
        # IMPORTANTE: Omni-3B solo para audio, NO para empatía
        if state.get("input_type") == "audio":
            print("🎙️ Ruta: Omni-3B Agent (audio español + NLLB)")
            return "omni"
        
        # PRIORIDAD 4: Expert para razonamiento técnico (SOLAR HTTP)
        if state.get("alpha", 0.0) > 0.7:
            print("🧠 Ruta: Expert Agent (SOLAR HTTP)")
            return "expert"
        
        # PRIORIDAD 5: Tiny (LFM2) para empathy + fallback
        # Maneja: soft > 0.7 (empatía), queries generales, fallback
        print("💬 Ruta: Tiny Agent (LFM2 empathy/fallback)")
        return "tiny"
    
    def _generate_expert(self, state: State) -> dict:
        """Nodo: Generar respuesta con expert agent"""
        print("🔬 Usando Expert Agent (SOLAR-10.7B)...")
        
        try:
            response = self.expert_agent.generate(
                state["input"],
                max_new_tokens=512
            )
        except Exception as e:
            print(f"⚠️  Error en expert agent: {e}")
            print("🔄 Fallback a tiny agent...")
            response = self.tiny_agent.generate(
                state["input"],
                soft_score=state["soft"],
                max_new_tokens=256
            )
        
        return {
            "agent_used": "expert",
            "response": response
        }
    
    def _generate_tiny(self, state: State) -> dict:
        """Nodo: Generar respuesta con tiny agent"""
        print("🏃 Usando Tiny Agent (LFM2-1.2B)...")
        
        response = self.tiny_agent.generate(
            state["input"],
            soft_score=state["soft"],
            max_new_tokens=256
        )
        
        return {
            "agent_used": "tiny",
            "response": response
        }
    
    def _execute_omni_loop(self, state: State) -> dict:
        """
        Nodo: Ejecutar Omni-Loop reflexivo (v2.16)
        
        Sistema de auto-corrección en 3 iteraciones:
        1. Draft inicial (skill_draft gRPC)
        2. Reflexión sobre draft
        3. Corrección final
        
        PHOENIX INTEGRATION:
        - skill_draft: Qwen3-VL-4B-Instruct (gRPC) - <500ms por iteración
        - skill_image: OpenCV + WebP + pHash (gRPC) - <100ms preprocesamiento
        - 0MB host RAM (procesamiento aislado)
        
        TARGET METRICS v2.16:
        - Latencia P50: <7.2s (3 iteraciones)
        - Auto-corrección: >71%
        - RAM P99: <9.6GB (containers aislados)
        - Confidence final: >0.85
        """
        print("🔄 Ejecutando Omni-Loop (reflexivo multimodal)...")
        
        try:
            # Detectar si hay imagen en el estado
            image_path = state.get("image_path")
            
            # Determinar si habilitar reflexión
            # Por defecto activada, pero puede deshabilitarse si soft > 0.8 (respuesta rápida emocional)
            enable_reflection = state.get("enable_reflection", True)
            if state.get("soft", 0.0) > 0.8:
                # Respuesta emocional rápida: deshabilitar reflexión
                enable_reflection = False
                print("⚡ Reflexión deshabilitada (respuesta emocional rápida)")
            
            # Ejecutar loop reflexivo
            result = self.omni_loop.execute_loop(
                prompt=state["input"],
                image_path=image_path,
                enable_reflection=enable_reflection
            )
            
            # Extraer respuesta y metadata
            response = result["response"]
            iterations = result["iterations"]
            auto_corrected = result["auto_corrected"]
            total_latency = result["total_latency_ms"]
            fallback_used = result["fallback_used"]
            
            # Logging de métricas
            print(f"✅ Omni-Loop completado en {total_latency:.1f}ms")
            print(f"   Iteraciones: {len(iterations)}")
            print(f"   Auto-corregido: {auto_corrected}")
            print(f"   Confidence final: {result['metadata']['confidence_final']:.2f}")
            
            if fallback_used:
                print(f"⚠️  Fallback usado: {result.get('fallback_reason', 'unknown')}")
            
            return {
                "agent_used": "omni_loop",
                "response": response,
                "omni_loop_iterations": iterations,
                "auto_corrected": auto_corrected
            }
        
        except Exception as e:
            print(f"❌ Error en Omni-Loop: {e}")
            print("🔄 Fallback a Tiny Agent...")
            
            # Fallback: usar tiny agent si loop falla completamente
            response = self.tiny_agent.generate(
                state["input"],
                soft_score=state.get("soft", 0.5),
                max_new_tokens=256
            )
            
            return {
                "agent_used": "tiny",
                "response": response,
                "omni_loop_iterations": [],
                "auto_corrected": False
            }
    
    def _generate_omni(self, state: State) -> dict:
        """Nodo: Generar respuesta con Omni-7B (v2.16)"""
        print("🌟 Usando Omni-7B Agent (LangChain)...")
        
        try:
            # Usar Omni-7B con LangChain
            # Contexto adicional basado en emoción detectada (si audio)
            query = state["input"]
            
            if state.get("detected_emotion"):
                emotion = state["detected_emotion"]
                # Prefijo de contexto emocional
                query = f"[Responde con tono {emotion}] {query}"
            
            response = self.omni_agent.invoke(
                query,
                max_tokens=512  # Balance entre calidad y velocidad
            )
            
            print(f"✅ Omni-7B respondió ({len(response)} chars)")
            
        except Exception as e:
            print(f"⚠️  Error en Omni-7B: {e}")
            print("🔄 Fallback a Tiny Agent...")
            response = self.tiny_agent.generate(
                state["input"],
                soft_score=state["soft"],
                max_new_tokens=256
            )
        
        return {
            "agent_used": "omni",
            "response": response
        }
    
    def _generate_vision(self, state: State) -> dict:
        """
        Nodo: Generar respuesta con Qwen3-VL-4B (v2.16.1)
        
        Best-of-Breed: Qwen3-VL-4B especializado en visión
        - MMMU: 60.1% (vs 59.2% Omni-7B)
        - MVBench: 71.9% (vs 70.3% Omni-7B)
        - VRAM: 3.3 GB (vs 4.9 GB Omni-7B)
        - TTL: 60s auto-unload
        """
        print("🖼️ Usando Qwen3-VL-4B Agent (Vision Specialist)...")
        
        try:
            # Lazy load del agente de visión
            vision_agent = get_qwen3_vl_agent()
            
            # Detectar tipo de input visual
            if state.get("input_type") == "image":
                response = vision_agent.invoke_vision(
                    prompt=state["input"],
                    image_path=state.get("image_path")
                )
                print(f"✅ Imagen analizada ({len(response)} chars)")
            
            elif state.get("input_type") == "video":
                response = vision_agent.invoke_vision(
                    prompt=state["input"],
                    video_path=state.get("video_path"),
                    fps=state.get("fps", 2.0)  # Default 2 FPS
                )
                print(f"✅ Video analizado ({len(response)} chars)")
            
            # Programar auto-unload después de 60s (TTL)
            from core.model_pool import get_model_pool
            model_pool = get_model_pool()
            model_pool.schedule_unload("qwen3_vl_4b", ttl=60)
            
        except Exception as e:
            print(f"⚠️  Error en Qwen3-VL-4B: {e}")
            print("🔄 Fallback a respuesta textual...")
            response = f"Lo siento, no pude procesar la {'imagen' if state.get('input_type') == 'image' else 'video'} debido a un error técnico."
        
        return {
            "agent_used": "vision",
            "response": response
        }
    
    def _log_feedback(self, state: State) -> dict:
        """Nodo: Registrar interacción y detectar feedback"""
        # Por ahora, feedback = 0 (se actualizará en interacción siguiente)
        feedback = 0.0
        
        self.feedback_detector.log_interaction(
            input_text=state["input"],
            hard=state["hard"],
            soft=state["soft"],
            alpha=state["alpha"],
            beta=state["beta"],
            agent_used=state["agent_used"],
            response=state["response"],
            feedback=feedback
        )
        
        # Agregar feedback al MCP
        self.mcp.add_feedback({
            "input": state["input"],
            "hard": state["hard"],
            "soft": state["soft"],
            "alpha": state["alpha"],
            "beta": state["beta"],
            "feedback": feedback
        })
        
        return {"feedback": feedback}
    
    # ============================================================
    # NODOS VOICE (v2.11)
    # ============================================================
    
    def _detect_input_type(self, state: State) -> dict:
        """
        Nodo: Detecta si el input es texto, audio o visión (v2.16.1)
        
        Returns:
            {"input_type": "text" | "audio" | "image" | "video"}
        """
        # PRIORIDAD 1: Visión (image o video)
        if state.get("image_path") or state.get("video_path"):
            if state.get("video_path"):
                print("🎥 Input detectado: VIDEO")
                return {"input_type": "video"}
            else:
                print("🖼️ Input detectado: IMAGEN")
                return {"input_type": "image"}
        
        # PRIORIDAD 2: Audio
        if state.get("audio_input"):
            print("🎤 Input detectado: AUDIO")
            return {"input_type": "audio"}
        
        # PRIORIDAD 3: Texto
        print("📝 Input detectado: TEXTO")
        return {"input_type": "text"}
    
    def _process_voice(self, state: State) -> dict:
        """
        Nodo: Pipeline completo de procesamiento de voz
        
        Pasos:
        1. Audio Router (Language ID)
        2. Omni-3B / NLLB / LFM2 (según idioma)
        3. STT + Detección de emoción
        4. Actualizar state con transcripción
        
        Returns:
            {
                "input": str,  # Transcripción
                "detected_emotion": str,
                "detected_lang": str,
                "voice_metadata": dict
            }
        """
        print("🎙️ Procesando input de voz...")
        
        try:
            from agents.audio_router import route_audio
            from agents.omni_pipeline import process_audio_input
            
            audio_bytes = state["audio_input"]
            
            # 1. Routing de audio (detecta idioma y selecciona motor)
            engine, audio, lang = route_audio(audio_bytes)
            print(f"🌐 Motor seleccionado: {engine}, Idioma: {lang}")
            
            # 2. Procesar según motor
            if engine == "omni":
                # Qwen3-VL-4B-Instruct: STT + Emoción nativos
                result = process_audio_input(audio_bytes)
                transcription = result.get("text", "")
                emotion = result.get("emotion", "neutral")
            
            elif engine == "nllb":
                # Pipeline de traducción (M3.1)
                from agents.nllb_translator import get_nllb_translator
                import whisper
                
                # STT con Whisper
                whisper_model = whisper.load_model("tiny")
                audio_result = whisper_model.transcribe(audio_bytes)
                text_original = audio_result["text"]
                
                # Traducir a español
                translator = get_nllb_translator()
                text_es = translator.translate(
                    text_original,
                    src_lang=lang,
                    tgt_lang="es"
                )
                
                transcription = text_es
                emotion = "neutral"  # NLLB no detecta emoción
            
            else:  # lfm2 fallback
                # Solo STT básico (sin emoción)
                import whisper
                whisper_model = whisper.load_model("tiny")
                result = whisper_model.transcribe(audio_bytes)
                transcription = result["text"]
                emotion = "neutral"
            
            print(f"📝 Transcripción: {transcription[:100]}...")
            print(f"😊 Emoción detectada: {emotion}")
            
            return {
                "input": transcription,
                "detected_emotion": emotion,
                "detected_lang": lang or "es",
                "voice_metadata": {
                    "engine": engine,
                    "original_lang": lang,
                    "stt_method": "omni" if engine == "omni" else "whisper"
                }
            }
        
        except Exception as e:
            print(f"⚠️ Error en process_voice: {e}")
            # SENTINEL: Fallback a texto vacío
            return {
                "input": "",
                "detected_emotion": "neutral",
                "detected_lang": "es",
                "voice_metadata": {"error": str(e)}
            }
    
    def _route_by_input_type(self, state: State) -> str:
        """Routing v2.16.1: vision → generate_vision, audio → process_voice, text → classify"""
        input_type = state.get("input_type")
        
        if input_type in ["image", "video"]:
            return "vision"
        elif input_type == "audio":
            return "audio"
        
        return "text"
    
    def _enhance_with_emotion(self, state: State) -> dict:
        """
        Nodo: Modula la respuesta según la emoción detectada (M3.2 Fase 2)
        
        Pipeline:
        1. Si detected_emotion existe → aplicar modulación
        2. Usar emotion_modulator para ajustar tono
        3. Retornar respuesta modulada
        """
        # Si no hay emoción detectada, pasar sin modificar
        if not state.get("detected_emotion"):
            return {"response": state["response"]}
        
        try:
            from agents.emotion_modulator import create_emotion_modulator
            
            print(f"😊 Modulando respuesta con emoción: {state['detected_emotion']}")
            
            # Obtener modulador
            modulator = create_emotion_modulator()
            
            # Modular respuesta
            modulated_response = modulator.modulate(
                text=state["response"],
                target_emotion=state["detected_emotion"]
            )
            
            print(f"✅ Modulación aplicada")
            
            return {"response": modulated_response}
        
        except Exception as e:
            print(f"⚠️ Error en modulación emocional: {e}")
            # SENTINEL: Modulación falla → usar respuesta original
            return {"response": state["response"]}
    
    def _generate_tts(self, state: State) -> dict:
        """
        Nodo: Genera audio de respuesta con TTS (M3.2 Fase 3)
        
        Pipeline:
        1. Extraer estado emocional del state
        2. Generar audio con TTSEngine (3-level fallback)
        3. Aplicar prosody según emoción
        4. Retornar audio_output
        """
        try:
            from agents.tts_engine import create_tts_engine
            
            print("🔊 Generando audio de respuesta...")
            
            # 1. Crear/obtener TTSEngine
            tts_engine = create_tts_engine()
            
            # 2. Preparar estado emocional (si disponible)
            emotion_state = None
            if state.get("detected_emotion"):
                try:
                    from agents.emotion_integration import EmotionState
                    
                    # Mapeo de emoción string → EmotionState
                    emotion_map = {
                        "empático": EmotionState(label="joy", valence=0.8, arousal=0.6, dominance=0.7),
                        "neutral": EmotionState(label="neutral", valence=0.5, arousal=0.5, dominance=0.5),
                        "urgente": EmotionState(label="anger", valence=0.2, arousal=0.9, dominance=0.8),
                    }
                    emotion_state = emotion_map.get(state["detected_emotion"])
                except ImportError:
                    print("⚠️ emotion_integration no disponible, TTS sin prosody")
            
            # 3. Generar audio
            tts_output = tts_engine.generate(
                text=state["response"],
                emotion_state=emotion_state
            )
            
            # 4. Logging
            print(f"✅ Audio generado: {tts_output.duration_ms}ms, "
                  f"Prosody: {tts_output.prosody_applied}, "
                  f"Cached: {tts_output.cached}")
            
            return {"audio_output": tts_output.audio_bytes}
        
        except Exception as e:
            print(f"⚠️ Error en TTS: {e}")
            # SENTINEL: TTS falla → continuar sin audio
            return {"audio_output": None}
    
    def _route_to_tts(self, state: State) -> str:
        """Routing: Si audio_input existe → TTS, sino → skip"""
        if state.get("audio_input"):
            return "tts"
        return "skip"
    
    # ============================================================
    # END NODOS VOICE
    # ============================================================
    
    def invoke(self, user_input: str) -> str:
        """
        Procesa input de TEXTO del usuario y retorna respuesta
        
        Args:
            user_input: Texto del usuario
        
        Returns:
            Respuesta generada
        """
        # Inicializar estado con campos por defecto (v2.11 con voice)
        initial_state = {
            "input": user_input,
            "input_type": "text",
            "audio_input": None,
            "audio_output": None,
            "detected_emotion": None,
            "detected_lang": None,
            "voice_metadata": {},
            "hard": 0.0,
            "soft": 0.0,
            "web_query": 0.0,
            "alpha": 0.5,
            "beta": 0.5,
            "agent_used": "tiny",
            "response": "",
            "feedback": 0.0,
            "rag_metadata": {}
        }
        
        result = self.app.invoke(initial_state)
        return result["response"]
    
    def invoke_audio(self, audio_bytes: bytes) -> dict:
        """
        Procesa input de AUDIO del usuario y retorna respuesta (v2.11)
        
        Args:
            audio_bytes: Audio crudo en bytes
        
        Returns:
            dict con:
                - response: str (texto de respuesta)
                - audio_output: bytes (audio TTS, si disponible)
                - detected_emotion: str
                - detected_lang: str
                - transcription: str
        """
        # Inicializar estado con audio (v2.11)
        initial_state = {
            "input": "",  # Se llenará por process_voice
            "input_type": "audio",
            "audio_input": audio_bytes,
            "audio_output": None,
            "detected_emotion": None,
            "detected_lang": None,
            "voice_metadata": {},
            "hard": 0.0,
            "soft": 0.0,
            "web_query": 0.0,
            "alpha": 0.5,
            "beta": 0.5,
            "agent_used": "tiny",
            "response": "",
            "feedback": 0.0,
            "rag_metadata": {}
        }
        
        result = self.app.invoke(initial_state)
        
        return {
            "response": result["response"],
            "audio_output": result.get("audio_output"),
            "detected_emotion": result.get("detected_emotion"),
            "detected_lang": result.get("detected_lang"),
            "transcription": result.get("input", ""),
            "voice_metadata": result.get("voice_metadata", {})
        }
    
    def invoke_multimodal(self, text: str, audio_path: str = None,
                         image_path: str = None) -> str:
        """
        Procesa input multimodal (v2.16: usa omni_native en lugar de multimodal_agent)
        
        Args:
            text: Texto del usuario
            audio_path: Ruta a archivo de audio (opcional)
            image_path: Ruta a imagen (opcional)
        
        Returns:
            Respuesta procesada
        """
        # Detectar si hay input multimodal (audio o imagen)
        has_multimodal = bool(audio_path or image_path)
        
        if has_multimodal:
            print("🎨 Detectado input multimodal")
            
            # v2.16.1: Omni-3B (audio) + Qwen3-VL (visión) especializados
            # Audio: Procesado con Omni-3B pipeline (ya en memoria)
            # Visión: Procesado con Qwen3-VL-4B (carga bajo demanda)
            
            if audio_path and not image_path:
                # Solo audio: procesar con Omni-3B pipeline
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                result = self.invoke_audio(audio_bytes)
                response = result["response"]
            
            elif image_path:
                # Imagen: usar preprocessor para descripción
                from core.image_preprocessor import ImagePreprocessor
                preprocessor = ImagePreprocessor()
                
                # Descripción textual de la imagen (placeholder hasta multimodal full)
                enhanced_text = f"{text}\n[Análisis de imagen: {image_path}]"
                response = self.omni_agent.invoke(enhanced_text, max_tokens=512)
            
            else:
                # Fallback a texto simple
                response = self.omni_agent.invoke(text, max_tokens=512)
            
            # Log como interacción multimodal
            self.feedback_detector.log_interaction(
                input_text=f"{text} [multimodal]",
                hard=0.5,
                soft=0.5,
                alpha=0.5,
                beta=0.5,
                agent_used="omni",  # v2.16: cambiado de "multimodal" a "omni"
                response=response,
                feedback=0.0
            )
            
            return response
        else:
            # Fallback a flujo normal
            return self.invoke(text)
    
    def get_statistics(self, days: int = 7) -> dict:
        """Obtiene estadísticas de rendimiento"""
        return self.feedback_detector.compute_statistics(days)


def create_orchestrator(use_simulated_trm: bool = False) -> SARAiOrchestrator:
    """
    Factory para crear orquestador
    
    Args:
        use_simulated_trm: True para usar TRM simulado (solo para testing)
                          False (default) usa TRM entrenado real (v2.11)
    """
    return SARAiOrchestrator(use_simulated_trm=use_simulated_trm)
