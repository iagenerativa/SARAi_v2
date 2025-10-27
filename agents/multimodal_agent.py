"""
Multimodal Agent para SARAi v2
Usa Qwen2.5-Omni-7B para procesamiento de audio y visión
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from typing import Optional, Union
import yaml
import gc
from pathlib import Path


class MultimodalAgent:
    """
    Agente multimodal basado en Qwen2.5-Omni-7B
    Maneja audio, visión y texto
    IMPORTANTE: Solo se carga cuando se detecta input multimodal
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['models']['multimodal']
        self.model = None
        self.processor = None
        self.device = torch.device("cpu")
        self.loaded = False
    
    def load(self):
        """Carga el modelo bajo demanda"""
        if self.loaded:
            return
        
        print(f"🔄 Cargando {self.config['name']} (~4GB, esto puede tardar)...")
        
        # Configuración de cuantización 4-bit para CPU
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.config['source'],
            cache_dir=self.config['cache_dir'],
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['source'],
            quantization_config=quantization_config,
            device_map="cpu",
            cache_dir=self.config['cache_dir'],
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        self.model.eval()
        self.loaded = True
        print(f"✅ {self.config['name']} cargado en CPU")
    
    def unload(self):
        """Descarga el modelo de memoria (CRÍTICO para liberar 4GB)"""
        if not self.loaded:
            return
        
        print(f"💾 Descargando {self.config['name']} de memoria...")
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self.loaded = False
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ 4GB liberados")
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio a texto
        
        Args:
            audio_path: Ruta al archivo de audio
        
        Returns:
            Transcripción del audio
        """
        if not self.loaded:
            self.load()
        
        # Procesar audio
        audio = self.processor(
            audios=audio_path,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **audio,
                max_new_tokens=256,
                do_sample=False
            )
        
        transcription = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Liberar memoria inmediatamente después de usar
        self.unload()
        
        return transcription.strip()
    
    def describe_image(self, image_path: str, question: Optional[str] = None) -> str:
        """
        Describe imagen o responde pregunta sobre ella
        
        Args:
            image_path: Ruta a la imagen
            question: Pregunta opcional sobre la imagen
        
        Returns:
            Descripción o respuesta
        """
        if not self.loaded:
            self.load()
        
        # Preparar prompt
        if question:
            prompt = f"<image>\n{question}"
        else:
            prompt = "<image>\nDescribe esta imagen en detalle."
        
        # Procesar imagen
        inputs = self.processor(
            text=prompt,
            images=image_path,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
        
        description = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Liberar memoria inmediatamente
        self.unload()
        
        return description.strip()
    
    def process_multimodal(self, text: str, audio_path: Optional[str] = None,
                          image_path: Optional[str] = None) -> str:
        """
        Procesa input multimodal combinado
        
        Args:
            text: Texto del usuario
            audio_path: Ruta a archivo de audio (opcional)
            image_path: Ruta a imagen (opcional)
        
        Returns:
            Respuesta procesando todas las modalidades
        """
        if not self.loaded:
            self.load()
        
        # Combinar modalidades
        full_input = text
        
        if audio_path:
            transcription = self.transcribe_audio(audio_path)
            full_input += f"\n[Audio transcrito: {transcription}]"
        
        if image_path:
            description = self.describe_image(image_path)
            full_input += f"\n[Imagen: {description}]"
        
        # Generar respuesta
        inputs = self.processor(
            text=full_input,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
                do_sample=True
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Liberar memoria
        self.unload()
        
        return response.strip()
    
    @staticmethod
    def detect_multimodal_input(input_data: dict) -> bool:
        """
        Detecta si el input requiere procesamiento multimodal
        
        Args:
            input_data: Dict con keys 'text', 'audio', 'image'
        
        Returns:
            True si hay audio o imagen
        """
        return bool(input_data.get('audio') or input_data.get('image'))


# Singleton global
_multimodal_agent = None

def get_multimodal_agent() -> MultimodalAgent:
    """Obtiene instancia singleton del multimodal agent"""
    global _multimodal_agent
    if _multimodal_agent is None:
        _multimodal_agent = MultimodalAgent()
    return _multimodal_agent
