"""
Health Dashboard v2.4 - SARAi
Endpoint con content negotiation:
- Accept: text/html → Dashboard interactivo para humanos
- Accept: application/json → JSON puro para monitoreo automatizado
"""

import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader

# Crear app FastAPI
app = FastAPI(
    title="SARAi Health Dashboard",
    description="Monitoreo de salud y KPIs del sistema SARAi v2.4",
    version="2.4.0"
)

# Template engine (si existe carpeta templates/)
templates_dir = Path(__file__).parent.parent / "templates"
if templates_dir.exists():
    templates = Environment(loader=FileSystemLoader(str(templates_dir)))
else:
    templates = None


def get_health_data() -> Dict[str, Any]:
    """
    Recopila métricas de salud del sistema
    
    Returns:
        Dict con KPIs y estado del sistema
    """
    # Métricas de sistema
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Verificar si hay modelos cargados (check cache dirs)
    models_loaded = []
    cache_dir = Path("models/cache")
    if cache_dir.exists():
        for subdir in cache_dir.iterdir():
            if subdir.is_dir() and any(subdir.iterdir()):
                models_loaded.append(subdir.name)
    
    # Estado del MCP (si existe state file)
    mcp_phase = 1  # Default
    mcp_state_path = Path("state/mcp_state.pkl")
    if mcp_state_path.exists():
        try:
            import pickle
            with open(mcp_state_path, 'rb') as f:
                mcp_state = pickle.load(f)
                mcp_phase = mcp_state.get('phase', 1)
        except:
            pass
    
    # Calcular cache hit rate (simulado, idealmente leer de logs)
    cache_hit_rate = 0.73  # Placeholder
    
    # Estado general
    ram_gb = memory.used / (1024 ** 3)
    status = "HEALTHY" if ram_gb <= 12.0 else "WARNING"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        
        # KPIs v2.4
        "ram_p99_gb": round(ram_gb, 2),
        "ram_percent": memory.percent,
        "cpu_percent": cpu_percent,
        "latency_p50_s": 25.4,  # Placeholder (leer de SARAi-Bench)
        "hard_accuracy": 0.87,  # Placeholder
        "empathy_score": 0.79,  # Placeholder
        
        # Estado del sistema
        "mcp_phase": mcp_phase,
        "models_loaded": models_loaded,
        "cache_hit_rate": cache_hit_rate,
        
        # Uptime
        "uptime_seconds": int(time.time() - psutil.boot_time()),
    }


@app.get("/health")
async def get_health(request: Request):
    """
    Health endpoint con content negotiation
    
    - Browser (Accept: text/html) → Devuelve dashboard HTML interactivo
    - curl/Docker (Accept: */*) → Devuelve JSON puro
    """
    health_data = get_health_data()
    
    # Content negotiation basada en header Accept
    accept_header = request.headers.get("accept", "")
    
    # Si el cliente pide HTML (navegador)
    if "text/html" in accept_header and templates:
        try:
            template = templates.get_template("health.html")
            html_content = template.render(
                data=health_data,
                kpis={
                    "RAM P99": {"value": health_data["ram_p99_gb"], "unit": "GB", "threshold": 12.0},
                    "Latency P50": {"value": health_data["latency_p50_s"], "unit": "s", "threshold": 30.0},
                    "Hard Accuracy": {"value": health_data["hard_accuracy"], "unit": "", "threshold": 0.85},
                    "Empathy": {"value": health_data["empathy_score"], "unit": "", "threshold": 0.75},
                }
            )
            return HTMLResponse(content=html_content)
        except Exception as e:
            # Si falla el template, devolver JSON
            return JSONResponse(content={
                **health_data,
                "template_error": str(e)
            })
    
    # Por defecto: JSON (para curl, Docker HEALTHCHECK, Prometheus)
    return JSONResponse(content=health_data)


@app.get("/")
async def root():
    """Redirect a /health"""
    return {"message": "SARAi v2.4 Health Dashboard", "endpoint": "/health"}


@app.get("/metrics")
async def metrics():
    """
    Endpoint en formato Prometheus con métricas completas
    
    Métricas expuestas:
    - sarai_ram_gb: Uso de RAM en GB (gauge)
    - sarai_cpu_percent: Uso de CPU (gauge)
    - sarai_response_latency_seconds: Histograma de latencia de respuestas
    - sarai_fallback_total: Contador de fallbacks por tipo (counter)
    - sarai_mcp_phase: Fase actual del MCP (gauge)
    - sarai_cache_hit_rate: Tasa de aciertos en cache (gauge)
    """
    health_data = get_health_data()
    
    # Leer fallbacks desde log
    fallback_counts = {}
    fallback_log = Path("state/model_fallbacks.log")
    if fallback_log.exists():
        try:
            import json
            with open(fallback_log, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    key = f"{entry['requested']}_to_{entry['used']}"
                    fallback_counts[key] = fallback_counts.get(key, 0) + 1
        except:
            pass
    
    # Formato Prometheus
    metrics_lines = [
        "# HELP sarai_ram_gb RAM usage in gigabytes",
        "# TYPE sarai_ram_gb gauge",
        f"sarai_ram_gb {health_data['ram_p99_gb']}",
        "",
        "# HELP sarai_cpu_percent CPU usage percentage",
        "# TYPE sarai_cpu_percent gauge",
        f"sarai_cpu_percent {health_data['cpu_percent']}",
        "",
        "# HELP sarai_response_latency_seconds Response latency P50 in seconds",
        "# TYPE sarai_response_latency_seconds gauge",
        f"sarai_response_latency_seconds {{quantile=\"0.5\"}} {health_data['latency_p50_s']}",
        "",
        "# HELP sarai_hard_accuracy Hard-intent classification accuracy",
        "# TYPE sarai_hard_accuracy gauge",
        f"sarai_hard_accuracy {health_data['hard_accuracy']}",
        "",
        "# HELP sarai_empathy_score Empathy score from soft-skills",
        "# TYPE sarai_empathy_score gauge",
        f"sarai_empathy_score {health_data['empathy_score']}",
        "",
        "# HELP sarai_mcp_phase MCP learning phase (1=rules, 2=MLP, 3=Transformer)",
        "# TYPE sarai_mcp_phase gauge",
        f"sarai_mcp_phase {health_data['mcp_phase']}",
        "",
        "# HELP sarai_cache_hit_rate Cache hit rate (0.0-1.0)",
        "# TYPE sarai_cache_hit_rate gauge",
        f"sarai_cache_hit_rate {health_data['cache_hit_rate']}",
        "",
        "# HELP sarai_fallback_total Total number of model fallbacks by type",
        "# TYPE sarai_fallback_total counter",
    ]
    
    # Añadir contadores de fallback
    for fallback_type, count in fallback_counts.items():
        requested, used = fallback_type.split("_to_")
        metrics_lines.append(
            f'sarai_fallback_total{{requested="{requested}",used="{used}"}} {count}'
        )
    
    if not fallback_counts:
        # Si no hay fallbacks, añadir línea vacía para evitar error en Prometheus
        metrics_lines.append('sarai_fallback_total{requested="none",used="none"} 0')
    
    metrics_lines.append("")
    
    # Métricas de uptime
    metrics_lines.extend([
        "# HELP sarai_uptime_seconds System uptime in seconds",
        "# TYPE sarai_uptime_seconds counter",
        f"sarai_uptime_seconds {health_data['uptime_seconds']}",
    ])
    
    metrics_text = "\n".join(metrics_lines)
    
    return HTMLResponse(content=metrics_text, media_type="text/plain")


# Para ejecutar con uvicorn:
# uvicorn sarai.health_dashboard:app --host 0.0.0.0 --port 8080 --reload

if __name__ == "__main__":
    import uvicorn
    print("🏥 Iniciando SARAi Health Dashboard v2.4...")
    print("Dashboard: http://localhost:8080/health")
    print("Metrics: http://localhost:8080/metrics")
    uvicorn.run(app, host="0.0.0.0", port=8080)
