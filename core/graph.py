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
from agents.expert_agent import get_expert_agent
from agents.tiny_agent import get_tiny_agent
from agents.multimodal_agent import get_multimodal_agent, MultimodalAgent
from agents.rag_agent import create_rag_node  # NEW v2.10


class State(TypedDict):
    """Estado compartido en el flujo LangGraph (v2.11 con Voice)"""
    # Input original
    input: str
    
    # Voice fields (v2.11)
    input_type: Literal["text", "audio"]
    audio_input: Optional[bytes]
    audio_output: Optional[bytes]
    detected_emotion: Optional[str]  # "empático" | "neutral" | "urgente"
    detected_lang: Optional[str]     # ISO 639-1 code
    voice_metadata: dict
    
    # Classification scores
    hard: float
    soft: float
    web_query: float  # v2.10
    
    # MCP weights
    alpha: float
    beta: float
    
    # Execution
    agent_used: Literal["expert", "tiny", "multimodal", "rag"]
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
        self.multimodal_agent = get_multimodal_agent()
        
        # NEW v2.10: RAG Agent (requiere model_pool)
        from core.model_pool import get_model_pool
        self.model_pool = get_model_pool()
        self.rag_node = create_rag_node(self.model_pool)
        
        # Construir grafo
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()
        
        print("✅ SARAi listo")
    
    def _build_graph(self) -> StateGraph:
        """Construye el grafo de estado de LangGraph (v2.11 con Voice)"""
        workflow = StateGraph(State)
        
        # Nodos (v2.11: añadidos detect_input_type y process_voice)
        workflow.add_node("detect_input_type", self._detect_input_type)
        workflow.add_node("process_voice", self._process_voice)
        workflow.add_node("classify", self._classify_intent)
        workflow.add_node("mcp", self._compute_weights)
        workflow.add_node("generate_expert", self._generate_expert)
        workflow.add_node("generate_tiny", self._generate_tiny)
        workflow.add_node("execute_rag", self.rag_node)
        workflow.add_node("enhance_with_emotion", self._enhance_with_emotion)  # v2.11
        workflow.add_node("generate_tts", self._generate_tts)  # v2.11
        workflow.add_node("feedback", self._log_feedback)
        
        # Flujo (v2.11: entrada por detect_input_type)
        workflow.set_entry_point("detect_input_type")
        
        # Routing de input: audio vs texto
        workflow.add_conditional_edges(
            "detect_input_type",
            self._route_by_input_type,
            {
                "audio": "process_voice",
                "text": "classify"
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
                "rag": "execute_rag"
            }
        )
        
        # LLMs → enhance_with_emotion
        workflow.add_edge("generate_expert", "enhance_with_emotion")
        workflow.add_edge("generate_tiny", "enhance_with_emotion")
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
        """Decisión de routing: rag > expert > tiny (v2.10)"""
        # PRIORIDAD 1: Si web_query > 0.7, usar RAG (v2.10)
        if state.get("web_query", 0.0) > 0.7:
            print("🌐 Ruta: RAG Agent (búsqueda web)")
            return "rag"
        
        # PRIORIDAD 2: Si alpha > 0.7, usar expert agent (SOLAR)
        if state["alpha"] > 0.7:
            print("🔬 Ruta: Expert Agent (SOLAR)")
            return "expert"
        
        # PRIORIDAD 3: Tiny agent (LFM2) por defecto
        print("🏃 Ruta: Tiny Agent (LFM2)")
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
        Nodo: Detecta si el input es texto o audio
        
        Returns:
            {"input_type": "text" | "audio"}
        """
        if state.get("audio_input"):
            print("🎤 Input detectado: AUDIO")
            return {"input_type": "audio"}
        
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
                # Qwen2.5-Omni-3B: STT + Emoción nativos
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
        """Routing: audio → process_voice, text → classify"""
        if state.get("input_type") == "audio":
            return "audio"
        return "text"
    
    def _enhance_with_emotion(self, state: State) -> dict:
        """
        Nodo: Modula la respuesta según la emoción detectada (v2.11 Fase 2)
        
        Por ahora, pasa la respuesta sin modificar.
        Implementación completa en Fase 2.
        """
        # TODO M3.2 Fase 2: Implementar modulación con LFM2
        return {"response": state["response"]}
    
    def _generate_tts(self, state: State) -> dict:
        """
        Nodo: Genera audio de respuesta con TTS (v2.11 Fase 3)
        
        Por ahora, retorna None.
        Implementación completa en Fase 3.
        """
        # TODO M3.2 Fase 3: Implementar TTS prosody-aware
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
        Procesa input multimodal
        
        Args:
            text: Texto del usuario
            audio_path: Ruta a archivo de audio (opcional)
            image_path: Ruta a imagen (opcional)
        
        Returns:
            Respuesta procesada
        """
        if MultimodalAgent.detect_multimodal_input({
            'audio': audio_path,
            'image': image_path
        }):
            print("🎨 Detectado input multimodal")
            response = self.multimodal_agent.process_multimodal(
                text, audio_path, image_path
            )
            
            # Log como interacción multimodal
            self.feedback_detector.log_interaction(
                input_text=f"{text} [multimodal]",
                hard=0.5,
                soft=0.5,
                alpha=0.5,
                beta=0.5,
                agent_used="multimodal",
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
