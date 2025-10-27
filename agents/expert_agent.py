"""
Expert Agent para SARAi v2
Usa SOLAR-10.7B-Instruct para tareas técnicas complejas
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional
import yaml
import gc


class ExpertAgent:
    """
    Agente experto basado en SOLAR-10.7B
    Especializado en hard-skills: código, matemáticas, lógica, configuración
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['models']['expert']
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        self.loaded = False
    
    def load(self):
        """Carga el modelo bajo demanda"""
        if self.loaded:
            return
        
        print(f"🔄 Cargando {self.config['name']} (~6GB, esto puede tardar)...")
        
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
        
        # Forzar liberación de memoria
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Memoria liberada")
    
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Genera respuesta técnica
        
        Args:
            prompt: Input del usuario
            max_new_tokens: Máximo de tokens a generar
        
        Returns:
            Respuesta generada
        """
        if not self.loaded:
            self.load()
        
        # Formato de prompt para SOLAR
        system_prompt = """Eres un asistente experto en programación, sistemas y tecnología.
Responde de forma técnica, precisa y concisa.
Incluye ejemplos de código cuando sea relevante.
Sé directo y enfócate en la solución."""
        
        full_prompt = f"""### System:
{system_prompt}

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
_expert_agent = None

def get_expert_agent() -> ExpertAgent:
    """Obtiene instancia singleton del expert agent"""
    global _expert_agent
    if _expert_agent is None:
        _expert_agent = ExpertAgent()
    return _expert_agent
