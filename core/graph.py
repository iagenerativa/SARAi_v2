"""
Orquestador LangGraph para SARAi v2
Flujo de estado: classify → mcp → route → generate → feedback
"""

from typing import TypedDict, Literal
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
    """Estado compartido en el flujo LangGraph (v2.10 con RAG)"""
    input: str
    hard: float
    soft: float
    web_query: float  # NEW v2.10
    alpha: float
    beta: float
    agent_used: Literal["expert", "tiny", "multimodal", "rag"]  # NEW v2.10
    response: str
    feedback: float
    rag_metadata: dict  # NEW v2.10: metadata de búsqueda web


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
        """Construye el grafo de estado de LangGraph (v2.10 con RAG)"""
        workflow = StateGraph(State)
        
        # Nodos
        workflow.add_node("classify", self._classify_intent)
        workflow.add_node("mcp", self._compute_weights)
        workflow.add_node("generate_expert", self._generate_expert)
        workflow.add_node("generate_tiny", self._generate_tiny)
        workflow.add_node("execute_rag", self.rag_node)  # NEW v2.10
        workflow.add_node("feedback", self._log_feedback)
        
        # Flujo
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "mcp")
        
        # Routing condicional basado en web_query (v2.10)
        workflow.add_conditional_edges(
            "mcp",
            self._route_to_agent,
            {
                "expert": "generate_expert",
                "tiny": "generate_tiny",
                "rag": "execute_rag"  # NEW v2.10
            }
        )
        
        workflow.add_edge("generate_expert", "feedback")
        workflow.add_edge("generate_tiny", "feedback")
        workflow.add_edge("execute_rag", "feedback")  # NEW v2.10
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
    
    def invoke(self, user_input: str) -> str:
        """
        Procesa input del usuario y retorna respuesta
        
        Args:
            user_input: Texto del usuario
        
        Returns:
            Respuesta generada
        """
        # Inicializar estado con campos por defecto (v2.10)
        initial_state = {
            "input": user_input,
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
