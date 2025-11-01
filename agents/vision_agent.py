"""
Vision Agent v2.12 - Análisis de imagen y video con Qwen3-VL-4B
Integración crítica para input multimodal

Capacidades:
- Análisis de imágenes (OCR, detección, descripción)
- Análisis de video (frame-by-frame, temporal)
- Visual reasoning (preguntas sobre contenido visual)
- Diagramas y gráficos

Benchmarks Qwen3-VL-4B (Q6_K):
- MMMU: 60.1% (visual understanding)
- MVBench: 71.9% (video analysis)
- Video-MME: 65.8% (video reasoning)
"""
import base64
from typing import Dict, Any, Optional, Union
from pathlib import Path
import io


class VisionAgent:
    """
    Agente de visión usando Qwen3-VL-4B (GGUF)
    
    Política de carga:
    - Bajo demanda cuando input_type in ["image", "video"]
    - TTL: 60s (auto-descarga rápida)
    - Auto-descarga si RAM libre < 4GB
    """
    
    def __init__(self, model_pool):
        """
        Args:
            model_pool: Instancia de ModelPool para gestión de modelos
        """
        self.model_pool = model_pool
        self.model_name = "qwen3_vl_4b"
    
    def analyze_image(
        self, 
        image_input: Union[str, bytes], 
        question: str = "¿Qué hay en esta imagen?",
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Analiza una imagen y responde una pregunta sobre ella
        
        Args:
            image_input: Ruta a imagen, bytes, o base64
            question: Pregunta sobre la imagen
            max_tokens: Máximo de tokens en la respuesta
        
        Returns:
            Dict con:
                - text: Respuesta textual
                - confidence: Score de confianza (si disponible)
                - metadata: Info adicional (tamaño imagen, etc.)
        
        Example:
            >>> agent = VisionAgent(model_pool)
            >>> result = agent.analyze_image("diagram.png", "Explica este diagrama")
            >>> print(result['text'])
        """
        # Cargar modelo (gestión automática de memoria por ModelPool)
        print(f"[VisionAgent] Cargando {self.model_name} para análisis de imagen...")
        model = self.model_pool.get(self.model_name)
        
        # Preparar imagen
        if isinstance(image_input, str):
            # Es una ruta
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"Imagen no encontrada: {image_input}")
            
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Codificar a base64 para el modelo
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        elif isinstance(image_input, bytes):
            # Ya son bytes
            image_b64 = base64.b64encode(image_input).decode('utf-8')
        
        else:
            # Asumir que ya es base64
            image_b64 = image_input
        
        # Construir prompt multimodal (formato Qwen-VL)
        prompt = f"""<|im_start|>system
You are a helpful vision assistant that analyzes images accurately.<|im_end|>
<|im_start|>user
<img>{image_b64}</img>
{question}<|im_end|>
<|im_start|>assistant
"""
        
        # Generar respuesta
        try:
            response = model.create_completion(
                prompt,
                max_tokens=max_tokens,
                temperature=0.4,  # Moderada para balance precisión/creatividad
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            
            response_text = response["choices"][0]["text"].strip()
            
            # Liberar modelo si es necesario (basado en RAM)
            self._auto_release_if_low_ram()
            
            return {
                "text": response_text,
                "confidence": None,  # GGUF no provee scores directamente
                "metadata": {
                    "model": self.model_name,
                    "question": question,
                    "tokens_generated": len(response_text.split())
                }
            }
        
        except Exception as e:
            # Asegurar liberación en caso de error
            self.model_pool.release(self.model_name)
            raise RuntimeError(f"Error en análisis de imagen: {e}")
    
    def analyze_video(
        self,
        video_path: str,
        question: str = "Describe qué sucede en este video",
        sample_fps: int = 1,  # Frames por segundo a analizar
        max_frames: int = 10
    ) -> Dict[str, Any]:
        """
        Analiza video frame-by-frame
        
        Args:
            video_path: Ruta al archivo de video
            question: Pregunta sobre el video
            sample_fps: FPS de muestreo (1 = 1 frame/segundo)
            max_frames: Máximo de frames a analizar
        
        Returns:
            Dict con análisis temporal del video
        """
        # TODO: Implementar extracción de frames con opencv
        # Por ahora, placeholder para la interfaz
        raise NotImplementedError(
            "Análisis de video requiere opencv-python. "
            "Instalar con: pip install opencv-python"
        )
    
    def _auto_release_if_low_ram(self):
        """
        Libera modelo automáticamente si RAM libre < 4GB
        Política defensiva para evitar OOM
        """
        import psutil
        
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_ram_gb < 4.0:
            print(f"[VisionAgent] RAM libre: {available_ram_gb:.1f}GB < 4GB. "
                  f"Liberando {self.model_name}...")
            self.model_pool.release(self.model_name)
    
    def describe_diagram(self, image_path: str) -> str:
        """
        Helper: Describe un diagrama técnico
        
        Args:
            image_path: Ruta a imagen de diagrama
        
        Returns:
            Descripción textual del diagrama
        """
        result = self.analyze_image(
            image_path,
            question="Describe este diagrama técnico en detalle. "
                     "Identifica componentes, conexiones y flujo de datos."
        )
        return result["text"]
    
    def extract_text_ocr(self, image_path: str) -> str:
        """
        Helper: Extrae texto de imagen (OCR)
        
        Args:
            image_path: Ruta a imagen con texto
        
        Returns:
            Texto extraído
        """
        result = self.analyze_image(
            image_path,
            question="Extrae todo el texto visible en esta imagen. "
                     "Mantén el formato y estructura."
        )
        return result["text"]


def create_vision_agent(model_pool) -> VisionAgent:
    """
    Factory para crear agente de visión
    
    Args:
        model_pool: Instancia de ModelPool
    
    Returns:
        VisionAgent configurado
    """
    return VisionAgent(model_pool)


# Exportaciones
__all__ = ["VisionAgent", "create_vision_agent"]
