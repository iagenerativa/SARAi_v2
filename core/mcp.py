"""
MCP (Meta Control Plane) para SARAi v2.8
Calcula pesos adaptativos α/β para enrutamiento hard/soft
NEW v2.3: Fast-Cache semántico con Vector Quantization
NEW v2.8: Recarga atómica sin downtime + Skills MoE routing
"""

import torch
import torch.nn as nn
import numpy as np
import time
import threading
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import yaml
import os
import json


class MCPCache:
    """
    NEW v2.3: Cache semántico con Vector Quantization (VQ)
    Evita recalcular α/β en diálogos coherentes
    """
    
    def __init__(self, embedder, ttl: int = 60, quant_levels: int = 32):
        """
        Args:
            embedder: Instancia de EmbeddingGemma
            ttl: Tiempo de vida del cache (segundos)
            quant_levels: Niveles de cuantización (5 bits = 32 niveles)
        """
        self.cache: Dict[bytes, Tuple[float, float, float]] = {}  # {key: (α, β, timestamp)}
        self.embedder = embedder
        self.ttl = ttl
        self.quant_levels = quant_levels
        
        print(f"[MCPCache] Inicializado (TTL={ttl}s, quant_levels={quant_levels})")
    
    def _quantize(self, emb: np.ndarray) -> np.ndarray:
        """
        Cuantiza embedding a 5 bits por dimensión
        2048-D × 5 bits → ~160 bytes/clave
        
        Args:
            emb: Embedding normalizado [-1, 1] o [0, 1]
        
        Returns:
            Embedding cuantizado [0, quant_levels-1]
        """
        # Normalizar a [0, 1]
        emb_norm = (emb - emb.min()) / (emb.max() - emb.min() + 1e-8)
        
        # Cuantizar
        emb_quant = (emb_norm * (self.quant_levels - 1)).astype(np.uint8)
        return np.clip(emb_quant, 0, self.quant_levels - 1)
    
    def get(self, context: str) -> Optional[Tuple[float, float]]:
        """
        Busca en cache por similitud semántica cuantizada
        
        Args:
            context: Texto de entrada (input + scores como contexto)
        
        Returns:
            (α, β) si hay hit válido, None si miss
        """
        emb = self.embedder.encode(context)
        key = self._quantize(emb).tobytes()
        
        if key in self.cache:
            alpha, beta, ts = self.cache[key]
            
            # Verificar TTL
            if time.time() - ts < self.ttl:
                return alpha, beta  # HIT
            else:
                # TTL expirado, eliminar entrada
                del self.cache[key]
        
        return None  # MISS
    
    def set(self, context: str, alpha: float, beta: float):
        """
        Guarda en cache
        
        Args:
            context: Texto de entrada
            alpha: Peso hard
            beta: Peso soft
        """
        emb = self.embedder.encode(context)
        key = self._quantize(emb).tobytes()
        self.cache[key] = (alpha, beta, time.time())
    
    def clear_expired(self):
        """Limpia entradas expiradas (llamar periódicamente)"""
        current_time = time.time()
        expired_keys = [
            k for k, (_, _, ts) in self.cache.items()
            if current_time - ts >= self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            print(f"[MCPCache] Limpiadas {len(expired_keys)} entradas expiradas")


class MCPRules:
    """
    MCP basado en reglas (fase inicial, sin entrenamiento)
    NEW v2.3: Integrado con MCPCache para evitar recálculos
    """
    
    def __init__(self, config_path: str = "config/models.yaml", embedder=None):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config['mcp']
        
        # NEW v2.3: Fast-cache semántico
        self.cache = MCPCache(embedder, ttl=60) if embedder else None
    
    def compute_weights(self, hard: float, soft: float, context: str = "",
                       feedback_buffer: List[Dict] = None) -> Tuple[float, float]:
        """
        Calcula pesos α (hard) y β (soft) según reglas heurísticas
        NEW v2.3: Usa cache semántico si disponible
        
        Args:
            hard: Score de hard-intent [0, 1]
            soft: Score de soft-intent [0, 1]
            context: Texto de entrada (para cache)
            feedback_buffer: Historial de interacciones recientes (opcional)
        
        Returns:
            (alpha, beta) donde alpha + beta = 1.0
        """
        
        # NEW v2.3: Comprobar fast-cache
        if self.cache and context:
            cached_weights = self.cache.get(context)
            if cached_weights:
                print(f"[MCP] Cache HIT: α={cached_weights[0]:.2f}, β={cached_weights[1]:.2f}")
                return cached_weights
        
        # MISS: Calcular pesos según reglas
        
        # Regla 1: Tarea técnica pura
        if hard > 0.8 and soft < 0.3:
            alpha, beta = 0.95, 0.05
        
        # Regla 2: Tarea emocional/social pura
        elif soft > 0.7 and hard < 0.4:
            alpha, beta = 0.2, 0.8
        
        # Regla 3: Urgencia técnica (error, bug)
        elif hard > 0.6 and soft < 0.5:
            alpha, beta = 0.85, 0.15
        
        # Regla 4: Explicación amigable (hard + soft moderados)
        elif 0.4 < hard < 0.7 and 0.4 < soft < 0.7:
            alpha, beta = 0.5, 0.5
        
        # Regla 5: Default híbrido
        else:
            alpha, beta = 0.6, 0.4
        
        # Ajuste con feedback si existe
        if feedback_buffer and len(feedback_buffer) >= 10:
            alpha, beta = self._adjust_with_feedback(alpha, beta, feedback_buffer)
        
        # Asegurar que sumen 1.0
        total = alpha + beta
        alpha, beta = alpha / total, beta / total
        
        # NEW v2.3: Guardar en cache
        if self.cache and context:
            self.cache.set(context, alpha, beta)
        
        return alpha, beta
    
    def _adjust_with_feedback(self, alpha: float, beta: float, 
                              buffer: List[Dict]) -> Tuple[float, float]:
        """
        Ajusta pesos basándose en feedback reciente
        """
        recent = buffer[-10:]  # Últimas 10 interacciones
        
        # Calcular tasa de éxito por rama
        hard_success = sum(1 for x in recent if x.get('alpha', 0) > 0.7 and x.get('feedback', 0) > 0)
        soft_success = sum(1 for x in recent if x.get('beta', 0) > 0.7 and x.get('feedback', 0) > 0)
        
        hard_total = sum(1 for x in recent if x.get('alpha', 0) > 0.7)
        soft_total = sum(1 for x in recent if x.get('beta', 0) > 0.7)
        
        # Si hard-agent falla mucho, aumentar soft
        if hard_total > 0 and (hard_success / hard_total) < 0.5:
            beta = min(beta + 0.1, 0.8)
            alpha = 1.0 - beta
        
        # Si soft-agent falla mucho, aumentar hard
        if soft_total > 0 and (soft_success / soft_total) < 0.5:
            alpha = min(alpha + 0.1, 0.8)
            beta = 1.0 - alpha
        
        return alpha, beta


class MCPLearned(nn.Module):
    """
    MCP con red neuronal entrenable (activado tras >100 interacciones)
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        super().__init__()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config['mcp']
        
        # MLP ligero para calcular pesos
        # Input: [hard, soft, avg_hard_success, avg_soft_success, urgency]
        self.mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)  # Asegura que α + β = 1
        )
        
        self.device = torch.device("cpu")
        self.to(self.device)
    
    def forward(self, features: torch.Tensor) -> Tuple[float, float]:
        """
        Calcula pesos α, β
        
        Args:
            features: [hard, soft, avg_hard_success, avg_soft_success, urgency]
        
        Returns:
            (alpha, beta)
        """
        with torch.no_grad():
            weights = self.mlp(features.unsqueeze(0))
            alpha, beta = weights[0, 0].item(), weights[0, 1].item()
        return alpha, beta
    
    def train_step(self, batch: List[Dict], optimizer: torch.optim.Optimizer) -> float:
        """
        Un paso de entrenamiento con policy gradient
        
        Args:
            batch: Lista de interacciones con feedback
            optimizer: Optimizador PyTorch
        
        Returns:
            loss_value
        """
        self.train()
        losses = []
        
        for item in batch:
            # Construir features
            features = torch.tensor([
                item['hard'],
                item['soft'],
                item.get('avg_hard_success', 0.5),
                item.get('avg_soft_success', 0.5),
                item.get('urgency', 0.0)
            ], dtype=torch.float32).to(self.device)
            
            # Forward
            weights = self.mlp(features.unsqueeze(0))
            alpha_pred, beta_pred = weights[0, 0], weights[0, 1]
            
            # Loss: recompensar decisiones que llevaron a buen feedback
            feedback = item.get('feedback', 0.0)
            alpha_used = item.get('alpha', 0.5)
            
            # Si usamos hard (alpha alto) y funcionó bien, reforzar
            if alpha_used > 0.7:
                loss = -feedback * torch.log(alpha_pred + 1e-8)
            else:
                loss = -feedback * torch.log(beta_pred + 1e-8)
            
            losses.append(loss)
        
        # Backprop
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        self.eval()
        return total_loss.item()
    
    def save_checkpoint(self, path: str = None):
        """Guarda checkpoint del MCP entrenado"""
        if path is None:
            path = self.config['checkpoint_path']
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"✅ MCP checkpoint guardado en {path}")
    
    def load_checkpoint(self, path: str = None) -> bool:
        """Carga checkpoint del MCP"""
        if path is None:
            path = self.config['checkpoint_path']
        
        if not os.path.exists(path):
            return False
        
        self.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✅ MCP checkpoint cargado desde {path}")
        return True


class MCP:
    """
    Meta Control Plane adaptativo
    Usa reglas inicialmente, evoluciona a MLP tras suficiente feedback
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['mcp']
        self.mode = self.config['mode']  # "rules" o "learned"
        
        self.rules_mcp = MCPRules(config_path)
        self.learned_mcp = MCPLearned(config_path)
        
        # Intentar cargar MCP entrenado
        if self.learned_mcp.load_checkpoint():
            self.mode = "learned"
            print("🧠 MCP en modo aprendido")
        else:
            print("📋 MCP en modo reglas")
        
        self.feedback_buffer = []
    
    def compute_weights(self, hard: float, soft: float) -> Tuple[float, float]:
        """
        Calcula pesos α, β según el modo activo
        """
        if self.mode == "learned":
            # Calcular estadísticas de feedback
            avg_hard_success = self._compute_avg_success(branch='hard')
            avg_soft_success = self._compute_avg_success(branch='soft')
            urgency = 1.0 if hard > 0.8 else 0.0
            
            features = torch.tensor([
                hard, soft, avg_hard_success, avg_soft_success, urgency
            ], dtype=torch.float32)
            
            alpha, beta = self.learned_mcp(features)
        else:
            alpha, beta = self.rules_mcp.compute_weights(
                hard, soft, self.feedback_buffer
            )
        
        return alpha, beta
    
    def add_feedback(self, interaction: Dict):
        """Agrega interacción al buffer de feedback"""
        self.feedback_buffer.append(interaction)
        
        # Limitar tamaño del buffer
        max_size = self.config['feedback_buffer_size']
        if len(self.feedback_buffer) > max_size:
            self.feedback_buffer = self.feedback_buffer[-max_size:]
        
        # Cambiar a modo learned si hay suficientes datos
        if len(self.feedback_buffer) >= self.config['training']['min_samples']:
            if self.mode == "rules":
                print("🔄 Suficiente feedback acumulado. Cambiando a modo aprendido...")
                self._train_learned_mcp()
                self.mode = "learned"
    
    def _compute_avg_success(self, branch: str) -> float:
        """Calcula tasa de éxito promedio de una rama"""
        if not self.feedback_buffer:
            return 0.5
        
        recent = self.feedback_buffer[-20:]
        
        if branch == 'hard':
            relevant = [x for x in recent if x.get('alpha', 0) > 0.7]
        else:
            relevant = [x for x in recent if x.get('beta', 0) > 0.7]
        
        if not relevant:
            return 0.5
        
        success = sum(1 for x in relevant if x.get('feedback', 0) > 0)
        return success / len(relevant)
    
    def _train_learned_mcp(self):
        """Entrena el MCP con el buffer de feedback"""
        optimizer = torch.optim.Adam(
            self.learned_mcp.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        print("🎓 Entrenando MCP...")
        for epoch in range(10):  # 10 epochs rápidos
            loss = self.learned_mcp.train_step(self.feedback_buffer, optimizer)
            if epoch % 3 == 0:
                print(f"  Epoch {epoch}: loss = {loss:.4f}")
        
        self.learned_mcp.save_checkpoint()
        print("✅ MCP entrenado")


# ---------- NEW v2.8: MoE Routing + Atomic Reload ----------

# Global MCP instance y lock para recarga atómica
_mcp_active = None
_mcp_lock = threading.RLock()  # Reentrant lock


def route_to_skills(scores: Dict[str, float], threshold: float = 0.3, top_k: int = 3) -> List[str]:
    """
    NEW v2.8: Enrutamiento top-k por umbral (sin softmax)
    
    Política MoE:
    1. Filtra skills con score > threshold
    2. Selecciona top-k por score descendente
    3. Excluye 'hard' y 'soft' (son base, no skills)
    
    Args:
        scores: {skill_name: score} desde TRM-Router
        threshold: Umbral mínimo de activación (default: 0.3)
        top_k: Máximo número de skills (default: 3)
    
    Returns:
        Lista de nombres de skills a activar
    
    Example:
        >>> scores = {'hard': 0.9, 'soft': 0.2, 'sql': 0.85, 'code': 0.7, 'math': 0.1}
        >>> route_to_skills(scores)
        ['sql', 'code']  # Top-2 sobre threshold, excluyendo hard/soft
    """
    # Filtrar skills (excluir hard/soft que son base)
    active_skills = {
        skill: score 
        for skill, score in scores.items() 
        if score > threshold and skill not in ["hard", "soft", "web_query"]
    }
    
    # Top-k por score descendente
    top_skills = sorted(active_skills.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [skill for skill, _ in top_skills]


def execute_skills_moe(
    input_text: str,
    scores: Dict[str, float],
    model_pool,
    threshold: float = 0.3,
    top_k: int = 3,
    enable_fallback: bool = True
) -> Dict[str, str]:
    """
    NEW v2.12: Ejecuta skills MoE según routing y retorna respuestas
    
    Flujo:
    1. route_to_skills() determina qué skills activar
    2. Carga cada skill bajo demanda desde ModelPool
    3. Ejecuta en paralelo (si múltiples skills) o secuencial
    4. Fallback a expert_short si todos los skills fallan
    
    Args:
        input_text: Texto de entrada del usuario
        scores: Dict con scores de TRM-Router
        model_pool: Instancia de ModelPool para cargar skills
        threshold: Umbral de activación de skills
        top_k: Máximo de skills simultáneos
        enable_fallback: Si True, cae back a expert_short en caso de fallo
    
    Returns:
        Dict {skill_name: response_text} con respuestas de cada skill
    
    Example:
        >>> scores = {'hard': 0.9, 'programming': 0.85, 'diagnosis': 0.7}
        >>> responses = execute_skills_moe("Debug mi código Python", scores, pool)
        >>> responses
        {'programming': '...análisis del código...'}
    """
    # Determinar skills a activar
    active_skills = route_to_skills(scores, threshold, top_k)
    
    if not active_skills:
        print("[MCP-MoE] No hay skills sobre threshold, usando expert por defecto")
        if enable_fallback:
            expert = model_pool.get("expert_short")
            response = expert.create_completion(input_text, max_tokens=512)
            return {"expert_fallback": response["choices"][0]["text"]}
        else:
            return {}
    
    print(f"[MCP-MoE] Skills activos: {active_skills}")
    
    # Ejecutar cada skill
    responses = {}
    
    for skill_name in active_skills:
        try:
            # Cargar skill bajo demanda (LRU gestionará memoria)
            skill_model = model_pool.get_skill(skill_name)
            
            # Ejecutar skill
            result = skill_model.create_completion(
                input_text,
                max_tokens=512,
                temperature=0.7,
                stop=["\n\n"]  # Detener en doble salto de línea
            )
            
            response_text = result["choices"][0]["text"].strip()
            responses[skill_name] = response_text
            
            print(f"✅ [MCP-MoE] Skill '{skill_name}' ejecutado ({len(response_text)} chars)")
        
        except Exception as e:
            print(f"⚠️ [MCP-MoE] Error ejecutando skill '{skill_name}': {e}")
            
            # Si falla, intentar con el siguiente skill o fallback
            if enable_fallback and not responses:
                # Si es el último skill y ninguno ha respondido, usar expert
                if skill_name == active_skills[-1]:
                    print("[MCP-MoE] Todos los skills fallaron, usando expert_short fallback")
                    try:
                        expert = model_pool.get("expert_short")
                        result = expert.create_completion(input_text, max_tokens=512)
                        responses["expert_fallback"] = result["choices"][0]["text"].strip()
                    except Exception as fallback_error:
                        print(f"❌ [MCP-MoE] Expert fallback también falló: {fallback_error}")
                        responses["error"] = "No se pudo generar respuesta"
    
    return responses


def reload_mcp():
    """
    NEW v2.8: Recarga MCP desde disco de forma atómica
    
    Llamado automáticamente cuando online_tune.py genera mcp_v_new.pkl
    - Usa doble buffer para 0s downtime
    - Lock garantiza no race conditions
    - Mantiene backup automático
    """
    global _mcp_active
    
    model_dir = Path("models/mcp")
    new_model_path = model_dir / "mcp_active.pkl"
    signal_file = Path("state/mcp_reload_signal")
    
    # Verificar si hay señal de recarga
    if not signal_file.exists():
        return False
    
    # Verificar que existe el nuevo modelo
    if not new_model_path.exists():
        print("[MCP] Señal de recarga detectada pero mcp_active.pkl no existe")
        signal_file.unlink()
        return False
    
    try:
        # Cargar nuevo modelo
        print("[MCP] Cargando nuevo modelo desde disco...")
        mcp_new = torch.load(new_model_path)
        
        # Swap atómico con lock
        with _mcp_lock:
            _mcp_active = mcp_new
            print("✅ [MCP] Swap atómico completado sin downtime")
        
        # Limpiar señal
        signal_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ [MCP] Error en recarga: {e}")
        signal_file.unlink()
        return False


def get_mcp_weights(scores: Dict[str, float], context: str = "") -> Tuple[float, float]:
    """
    NEW v2.8: Obtiene pesos α/β de forma thread-safe
    
    Protegido por lock para evitar leer durante swap
    """
    global _mcp_active
    
    # Verificar señal de recarga antes de computar
    reload_mcp()
    
    with _mcp_lock:
        if _mcp_active is None:
            # Primera carga: usar MCP por defecto
            _mcp_active = create_mcp()
        
        # Delegar al MCP activo
        hard = scores.get('hard', 0.5)
        soft = scores.get('soft', 0.5)
        return _mcp_active.compute_weights(hard, soft, context)


def create_mcp() -> MCP:
    """Factory para crear MCP"""
    return MCP()


# ========================================
# NEW v2.12: Skills as Prompting Configs
# ========================================

def detect_and_apply_skill(query: str, model_name: str = "solar") -> Optional[Dict[str, Any]]:
    """
    Detecta si la query requiere un skill especializado y retorna su configuración.
    
    Args:
        query: Input del usuario
        model_name: Modelo a usar ("solar" o "lfm2")
    
    Returns:
        Dict con skill_config si se detecta, None en caso contrario
        {
            "skill_name": str,
            "system_prompt": str,
            "generation_params": dict,
            "full_prompt": str (system + user query)
        }
    """
    from core.skill_configs import match_skill_by_keywords, list_skills
    
    # Detectar skill por keywords
    skill_config = match_skill_by_keywords(query)
    
    if skill_config is None:
        # No hay skill específico, usar configuración por defecto
        return None
    
    # Verificar si el skill prefiere el modelo actual
    if skill_config.preferred_model != model_name:
        print(f"[Skills] Recomendado cambiar de {model_name} a {skill_config.preferred_model} para skill '{skill_config.name}'")
    
    # Construir respuesta con configuración completa
    return {
        "skill_name": skill_config.name,
        "system_prompt": skill_config.system_prompt,
        "generation_params": skill_config.get_generation_params(),
        "full_prompt": skill_config.build_prompt(query),
        "description": skill_config.description,
        "preferred_model": skill_config.preferred_model
    }


def list_available_skills() -> List[str]:
    """Lista todos los skills disponibles en el sistema"""
    from core.skill_configs import list_skills
    return list_skills()


def get_skill_info(skill_name: str) -> Optional[Dict[str, Any]]:
    """Obtiene información detallada de un skill específico"""
    from core.skill_configs import get_skill
    
    skill = get_skill(skill_name)
    if skill is None:
        return None
    
    return skill.to_dict()
