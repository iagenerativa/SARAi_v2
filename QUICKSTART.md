# 🚀 SARAi v2 - Inicio Rápido

## Instalación en 3 pasos

```bash
# 1. Ejecutar instalador
bash install.sh

# 2. Activar entorno
source venv/bin/activate

# 3. Iniciar SARAi
python main.py
```

## Primera vez (Modo Simulado)

El sistema usa TRM-Classifier **simulado** basado en keywords. Esto permite probar el sistema inmediatamente sin entrenar modelos.

```bash
python main.py
```

Ejemplo:
```
Tú: ¿Cómo configuro SSH en Linux?
📊 Intent: hard=0.92, soft=0.15
⚖️  Pesos: α=0.95 (hard), β=0.05 (soft)
🔬 Usando Expert Agent (SOLAR-10.7B)...
```

## Entrenar TRM Real (Opcional)

Para mejor clasificación, entrena el TRM:

```bash
# 1. Generar dataset
python scripts/generate_synthetic_data.py --samples 1000

# 2. Entrenar
python scripts/train_trm.py --data data/trm_training.json --epochs 50

# 3. Usar TRM entrenado
python main.py --use-real-trm
```

## Comandos Útiles

```bash
# Ver estadísticas de rendimiento
python main.py --stats

# Ver estadísticas de últimos 30 días
python main.py --stats --days 30

# Ejecutar tests
python -m pytest tests/ -v

# Limpiar logs antiguos
find logs/ -name "*.jsonl" -mtime +30 -delete
```

## Gestión de Memoria

SARAi gestiona memoria automáticamente, pero puedes optimizar:

**Para sistemas con <12GB RAM disponible:**

Edita `config/models.yaml`:
```yaml
memory:
  max_concurrent_llms: 1  # Solo 1 LLM en RAM a la vez
```

**Para acelerar respuestas (sacrificando memoria):**

```yaml
memory:
  unload_timeout_seconds: 300  # Mantener modelos 5 min en RAM
```

## Estructura Simplificada

```
SARAi_v2/
├── main.py              ← Punto de entrada
├── config/models.yaml   ← Configuración
├── core/                ← Componentes principales
├── agents/              ← Agentes LLM
├── logs/                ← Historial de interacciones
└── README.md            ← Documentación completa
```

## Troubleshooting Rápido

### Error: "Out of memory"
```bash
# Reducir carga en config/models.yaml
memory:
  max_concurrent_llms: 1
```

### Los modelos se descargan lentamente
```bash
# Usar mirror de HuggingFace
export HF_ENDPOINT=https://hf-mirror.com
python main.py
```

### Quiero respuestas más rápidas
Usa solo el tiny agent editando `core/graph.py` línea 89:
```python
if state["alpha"] > 0.95:  # Solo expert en casos extremos
```

## Próximos Pasos

1. ✅ **Prueba el sistema** con `python main.py`
2. 📊 **Acumula interacciones** (el sistema aprende del feedback)
3. 🎓 **Entrena el TRM** tras ~100 interacciones
4. 🧠 **Activa modo learned** en el MCP

---

**¿Necesitas ayuda?** Consulta `README.md` para documentación completa.
