"""
Tiny Agent para SARAi v2
Usa LFM2-1.2B para respuestas rápidas y soft-skills
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional
import yaml
import gc


class TinyAgent:
    """
    Agente ligero basado en LFM2-1.2B
    Especializado en respuestas rápidas y soft-skills con modulación de estilo
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['models']['tiny']
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        self.loaded = False
    
    def load(self):
        """Carga el modelo bajo demanda"""
        if self.loaded:
            return
        
        print(f"🔄 Cargando {self.config['name']} (~700MB)...")
        
        # Configuración de cuantización 4-bit para CPU
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
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
        """Descarga el modelo de memoria"""
        if not self.loaded:
            return
        
        print(f"💾 Descargando {self.config['name']} de memoria...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Memoria liberada")
    
    def get_style_prompt(self, soft_score: float) -> str:
        """
        Genera instrucciones de estilo según soft-score
        
        Args:
            soft_score: Score de soft-intent [0, 1]
        
        Returns:
            Instrucciones de sistema
        """
        if soft_score > 0.7:
            return """Responde con empatía, calidez y comprensión emocional.
Demuestra que entiendes los sentimientos del usuario.
Usa un tono amable y cercano.
Ofrece apoyo y ánimo cuando sea apropiado."""
        
        elif soft_score > 0.4:
            return """Responde de forma amigable y accesible.
Usa un lenguaje claro y ejemplos comprensibles.
Sé paciente y explicativo.
Mantén un tono conversacional y positivo."""
        
        else:
            return """Responde de forma neutral y directa.
Sé claro y conciso.
Enfócate en la información útil.
Mantén un tono profesional pero accesible."""
    
    def generate(self, prompt: str, soft_score: float = 0.5, 
                max_new_tokens: int = 256) -> str:
        """
        Genera respuesta con modulación de estilo
        
        Args:
            prompt: Input del usuario
            soft_score: Score de soft-intent para modular estilo
            max_new_tokens: Máximo de tokens a generar
        
        Returns:
            Respuesta generada
        """
        if not self.loaded:
            self.load()
        
        # Obtener instrucciones de estilo
        style_prompt = self.get_style_prompt(soft_score)
        
        # Formato de prompt
        full_prompt = f"""### System:
{style_prompt}

### User:
{prompt}

### Assistant:
"""
        
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['context_length']
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config['temperature'],
                top_p=self.config['top_p'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


# Singleton global
_tiny_agent = None

def get_tiny_agent() -> TinyAgent:
    """Obtiene instancia singleton del tiny agent"""
    global _tiny_agent
    if _tiny_agent is None:
        _tiny_agent = TinyAgent()
    return _tiny_agent
