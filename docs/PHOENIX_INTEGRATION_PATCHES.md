# 🔧 Phoenix Integration Patches (Ready-to-Deploy)

**Objetivo**: Integrar Skills-as-Services (Phoenix v2.12) en el roadmap v2.15 con **copy-paste directo**.

**Timeline**: Nov 8 → Nov 25 (17 días, 3 fases)

---

## 📦 Patch #1: ModelPool.get_skill_client() (3 líneas)

**Archivo**: `core/model_pool.py`  
**Tiempo**: 5 min  
**Testing**: `test_model_pool_skill_client.py`

```python
# core/model_pool.py - Añadir al final de la clase ModelPool

def get_skill_client(self, skill_name: str):
    """
    Launch skill container and return gRPC client
    
    Args:
        skill_name: Nombre del skill (sql, draft, patch-sandbox, etc.)
    
    Returns:
        SkillServiceStub: Cliente gRPC listo para Infer()
    
    Raises:
        RuntimeError: Si el container falla al iniciar
    """
    import subprocess
    import grpc
    from skills import skills_pb2_grpc
    
    # Cache de clientes activos
    if not hasattr(self, '_skill_clients'):
        self._skill_clients = {}
    
    if skill_name not in self._skill_clients:
        # 1. Verificar si container ya existe (reinicio de SARAi)
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name=saraiskill.{skill_name}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        
        if "Up" in result.stdout:
            logger.info(f"✅ Skill {skill_name} ya activo, reutilizando container")
        else:
            # 2. Lanzar nuevo container
            logger.info(f"🚀 Lanzando skill container: {skill_name}")
            
            try:
                subprocess.run([
                    "docker", "run", "-d",
                    "--name", f"saraiskill.{skill_name}",
                    "--cap-drop=ALL",
                    "--read-only",
                    "--tmpfs", "/tmp:size=256M",
                    "--security-opt", "no-new-privileges:true",
                    "-p", "50051:50051",  # TODO: Dynamic port allocation
                    f"saraiskill.{skill_name}:v2.12"
                ], check=True, capture_output=True, timeout=30)
                
                # Esperar health check
                time.sleep(2)
                
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Skill {skill_name} falló al iniciar: {e.stderr}")
                raise RuntimeError(f"Skill {skill_name} launch failed")
        
        # 3. Crear cliente gRPC
        channel = grpc.insecure_channel('localhost:50051')
        self._skill_clients[skill_name] = skills_pb2_grpc.SkillServiceStub(channel)
        
        logger.info(f"✅ Skill {skill_name} client ready")
    
    return self._skill_clients[skill_name]
```

**Validación**:
```python
# tests/test_model_pool_skill_client.py
def test_get_skill_client_launches_container():
    pool = ModelPool(config)
    client = pool.get_skill_client("sql")
    
    # Verificar container running
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=saraiskill.sql", "--format", "{{.Status}}"],
        capture_output=True,
        text=True
    )
    assert "Up" in result.stdout
    
    # Verificar gRPC health check
    request = HealthCheckRequest()
    response = client.Check(request)
    assert response.status == HealthCheckResponse.SERVING

def test_get_skill_client_reuses_existing():
    pool = ModelPool(config)
    client1 = pool.get_skill_client("sql")
    client2 = pool.get_skill_client("sql")
    
    # Mismo objeto (no lanza container duplicado)
    assert client1 is client2
```

---

## 📦 Patch #2: docker-compose.sentience.yml (Skills Services)

**Archivo**: `docker-compose.sentience.yml`  
**Tiempo**: 10 min  
**Testing**: `docker-compose config` (validar sintaxis)

```yaml
version: '3.8'

services:
  # Servicio principal SARAi (sin cambios)
  sarai:
    image: sarai/omni-sentinel:2.15
    container_name: sarai-sentience
    # ... configuración existente ...
  
  # ========================================
  # PHOENIX SKILLS (NEW v2.12)
  # ========================================
  
  skill-sql:
    image: saraiskill.sql:v2.12
    container_name: saraiskill.sql
    
    # Hardening Phoenix (heredado)
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    
    tmpfs:
      - /tmp:size=256M,mode=1777
    
    # Networking
    ports:
      - "50051:50051"
    networks:
      - sarai_net
    
    # Health check gRPC
    healthcheck:
      test: ["CMD", "grpc_health_probe", "-addr=localhost:50051"]
      interval: 15s
      timeout: 2s
      retries: 3
      start_period: 5s
    
    # Recursos (v2.13 EntityMemory)
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
  
  skill-draft:
    image: saraiskill.draft:v2.14
    container_name: saraiskill.draft
    
    # Hardening idéntico a skill-sql
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp:size=256M,mode=1777
    
    # Port diferente (multi-skill)
    ports:
      - "50052:50051"
    networks:
      - sarai_net
    
    healthcheck:
      test: ["CMD", "grpc_health_probe", "-addr=localhost:50051"]
      interval: 15s
      timeout: 2s
      retries: 3
    
    # Recursos (v2.14 Speculative Decode)
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 800M  # Qwen2.5-0.5B IQ2 ~650MB
        reservations:
          cpus: '1'
          memory: 400M
  
  # Servicios auxiliares (sin cambios)
  prometheus:
    image: prom/prometheus:latest
    # ... configuración existente ...
  
  grafana:
    image: grafana/grafana:latest
    # ... configuración existente ...

networks:
  sarai_net:
    driver: bridge
```

**Validación**:
```bash
# 1. Validar sintaxis
docker-compose -f docker-compose.sentience.yml config

# 2. Levantar skills
docker-compose -f docker-compose.sentience.yml up -d skill-sql skill-draft

# 3. Verificar health checks
docker ps --filter "name=saraiskill" --format "table {{.Names}}\t{{.Status}}"

# Expected output:
# NAMES                   STATUS
# saraiskill.sql         Up 10 seconds (healthy)
# saraiskill.draft       Up 8 seconds (healthy)
```

---

## 📦 Patch #3: EntityMemory con skill_sql (v2.13)

**Archivo**: `core/entity_memory.py`  
**Tiempo**: 20 min  
**Testing**: `test_entity_memory_skill_integration.py`

```python
# core/entity_memory.py - Modificar clase EntityMemory

class EntityMemory:
    """
    Persistent entity storage usando skill_sql (Phoenix-optimized)
    
    CAMBIO v2.13: Queries ejecutadas en container SQL aislado
    """
    
    def __init__(self, use_skill: bool = True):
        """
        Args:
            use_skill: Si True, usa skill_sql container (default v2.13)
                       Si False, usa SQLite local (fallback)
        """
        self.use_skill = use_skill
        
        if self.use_skill:
            from core.model_pool import get_model_pool
            pool = get_model_pool()
            
            try:
                self.sql_client = pool.get_skill_client("sql")
                logger.info("✅ EntityMemory usando skill_sql container")
            except RuntimeError:
                logger.warning("⚠️ skill_sql no disponible, fallback a SQLite local")
                self.use_skill = False
        
        if not self.use_skill:
            # Fallback a SQLite local
            import sqlite3
            self.conn = sqlite3.connect("state/entity_memory.db")
            self._init_schema()
    
    def store_svo(self, subject: str, verb: str, obj: str, confidence: float = 1.0):
        """
        Almacena triple SVO
        
        CAMBIO v2.13: Query ejecutada via gRPC si use_skill=True
        """
        query = f"""
        INSERT INTO entity_memory (subject, verb, object, confidence, source)
        VALUES ('{self._escape(subject)}', '{self._escape(verb)}', 
                '{self._escape(obj)}', {confidence}, 'user_input')
        """
        
        if self.use_skill:
            # gRPC call a skill_sql
            from skills import skills_pb2
            
            request = skills_pb2.InferRequest(
                prompt=query,
                max_tokens=0,  # No generación, solo execute
                temperature=0.0
            )
            
            response = self.sql_client.Infer(request, timeout=5.0)
            
            if response.metrics.error:
                logger.error(f"skill_sql error: {response.metrics.error}")
                raise RuntimeError("SQL execution failed")
            
            logger.debug(f"Stored SVO via skill: ({subject}, {verb}, {obj})")
        
        else:
            # Fallback SQLite local
            self.conn.execute(query)
            self.conn.commit()
            logger.debug(f"Stored SVO locally: ({subject}, {verb}, {obj})")
    
    def recall(self, subject: str = None, verb: str = None, obj: str = None) -> List[Triple]:
        """
        Busca triples por S/V/O (query SVO optimizada)
        
        CAMBIO v2.13: Usa índice triple en skill_sql
        """
        conditions = []
        if subject:
            conditions.append(f"subject = '{self._escape(subject)}'")
        if verb:
            conditions.append(f"verb = '{self._escape(verb)}'")
        if obj:
            conditions.append(f"object = '{self._escape(obj)}'")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
        SELECT subject, verb, object, confidence, timestamp
        FROM entity_memory
        WHERE {where_clause}
        ORDER BY confidence DESC, timestamp DESC
        LIMIT 100
        """
        
        if self.use_skill:
            from skills import skills_pb2
            
            request = skills_pb2.InferRequest(
                prompt=query,
                max_tokens=2048,  # Resultados pueden ser largos
                temperature=0.0
            )
            
            response = self.sql_client.Infer(request, timeout=10.0)
            
            # Parse response (assume JSON format)
            import json
            results = json.loads(response.text)
            
            return [
                Triple(
                    subject=row['subject'],
                    verb=row['verb'],
                    object=row['object'],
                    confidence=row['confidence']
                )
                for row in results
            ]
        
        else:
            # Fallback SQLite local
            cursor = self.conn.execute(query)
            return [
                Triple(
                    subject=row[0],
                    verb=row[1],
                    object=row[2],
                    confidence=row[3]
                )
                for row in cursor.fetchall()
            ]
    
    def _escape(self, text: str) -> str:
        """Escapa comillas simples para SQL"""
        return text.replace("'", "''")
```

**Validación**:
```python
# tests/test_entity_memory_skill_integration.py
def test_store_svo_via_skill():
    memory = EntityMemory(use_skill=True)
    memory.store_svo("hermana", "vive_en", "Madrid", confidence=0.9)
    
    # Verificar en DB del container
    triples = memory.recall(subject="hermana")
    assert len(triples) == 1
    assert triples[0].object == "Madrid"

def test_fallback_to_local_if_skill_unavailable():
    # Detener skill_sql container
    subprocess.run(["docker", "stop", "saraiskill.sql"])
    
    memory = EntityMemory(use_skill=True)
    # Debe caer a SQLite local automáticamente
    assert memory.use_skill == False
    
    # Funciona igual
    memory.store_svo("test", "is", "working")
    assert len(memory.recall(subject="test")) == 1
```

---

## 📦 Patch #4: Grafana Panel "Skill Hit Rate" (JSON)

**Archivo**: `extras/grafana_phoenix_skills.json`  
**Tiempo**: 5 min  
**Import**: Grafana UI → Dashboards → Import → Paste JSON

```json
{
  "dashboard": {
    "title": "Phoenix Skills-as-Services v2.12",
    "uid": "phoenix-skills-v2-12",
    "tags": ["sarai", "phoenix", "skills"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Skill Hit Rate (%)",
        "type": "graph",
        "targets": [
          {
            "expr": "(rate(skill_requests_total[5m]) / rate(sarai_requests_total[5m])) * 100",
            "legendFormat": "Skill Usage %"
          }
        ],
        "yaxes": [
          {"format": "percent", "min": 0, "max": 100}
        ],
        "alert": {
          "conditions": [
            {
              "evaluator": {"type": "lt", "params": [40]},
              "query": {"params": ["A", "5m", "now"]},
              "type": "query"
            }
          ],
          "message": "Skill hit rate below 40% - check MoE routing"
        }
      },
      {
        "id": 2,
        "title": "Skill Latency P50 (ms)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.5, rate(skill_latency_seconds_bucket[5m])) * 1000",
            "legendFormat": "{{skill_name}}"
          }
        ],
        "yaxes": [
          {"format": "ms", "min": 0}
        ]
      },
      {
        "id": 3,
        "title": "Skill Container Health",
        "type": "table",
        "targets": [
          {
            "expr": "up{job='skill-containers'}",
            "format": "table",
            "instant": true
          }
        ],
        "transformations": [
          {
            "id": "organize",
            "options": {
              "excludeByName": {},
              "indexByName": {"skill_name": 0, "status": 1},
              "renameByName": {"skill_name": "Skill", "status": "Health"}
            }
          }
        ]
      },
      {
        "id": 4,
        "title": "RAM per Skill Container (MB)",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes{name=~'saraiskill.*'} / 1024 / 1024",
            "legendFormat": "{{name}}"
          }
        ],
        "yaxes": [
          {"format": "decmbytes", "min": 0, "max": 1024}
        ]
      },
      {
        "id": 5,
        "title": "Speculative Decode Acceptance Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "sarai_speculative_acceptance_rate"
          }
        ],
        "options": {
          "showThresholdLabels": false,
          "showThresholdMarkers": true
        },
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"value": 0, "color": "red"},
                {"value": 0.6, "color": "yellow"},
                {"value": 0.8, "color": "green"}
              ]
            },
            "min": 0,
            "max": 1
          }
        }
      },
      {
        "id": 6,
        "title": "Federated Learning Privacy (ε)",
        "type": "stat",
        "targets": [
          {
            "expr": "sarai_fl_epsilon"
          }
        ],
        "options": {
          "colorMode": "value",
          "graphMode": "none",
          "textMode": "value_and_name"
        },
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 1.0, "color": "yellow"},
                {"value": 2.0, "color": "red"}
              ]
            }
          }
        }
      }
    ],
    "time": {
      "from": "now-6h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

**Métricas requeridas** (añadir a `sarai/health_dashboard.py`):
```python
# sarai/health_dashboard.py - Añadir al final

# Phoenix Skills metrics
skill_requests_total = Counter('skill_requests_total', 'Total requests to skills', ['skill_name'])
skill_latency_seconds = Histogram('skill_latency_seconds', 'Skill inference latency', ['skill_name'])
skill_up = Gauge('up', 'Skill container health', ['job', 'skill_name'])

# Speculative decoding (v2.14)
speculative_acceptance_rate = Gauge('sarai_speculative_acceptance_rate', 'Draft token acceptance rate')

# Federated Learning (v2.15)
fl_epsilon = Gauge('sarai_fl_epsilon', 'Differential privacy epsilon')
```

---

## 🚀 Quick Deploy (All Patches)

**Script completo** (ejecutar desde raíz del repo):

```bash
#!/bin/bash
# deploy_phoenix_integration.sh

set -e

echo "🔥 Phoenix Integration Deployment"
echo "=================================="

# 1. Build skill images
echo "📦 Step 1/5: Building skill images..."
make skill-image SKILL=sql
make skill-image SKILL=draft  # Para v2.14
echo "✅ Skill images built"

# 2. Apply ModelPool patch
echo "📦 Step 2/5: Patching ModelPool..."
# (Aplicar manualmente patch #1 o usar sed)
echo "⚠️  MANUAL: Apply Patch #1 to core/model_pool.py"

# 3. Update docker-compose
echo "📦 Step 3/5: Updating docker-compose..."
cp docker-compose.sentience.yml docker-compose.sentience.yml.backup
# (Aplicar manualmente patch #2)
echo "⚠️  MANUAL: Apply Patch #2 to docker-compose.sentience.yml"

# 4. Start skill containers
echo "📦 Step 4/5: Starting skill containers..."
docker-compose -f docker-compose.sentience.yml up -d skill-sql skill-draft

# 5. Validate health
echo "📦 Step 5/5: Validating health checks..."
sleep 5
docker ps --filter "name=saraiskill" --format "table {{.Names}}\t{{.Status}}"

echo ""
echo "✅ Phoenix Integration deployed successfully!"
echo ""
echo "Next steps:"
echo "  1. Apply Patch #3 (EntityMemory) for v2.13"
echo "  2. Import Patch #4 (Grafana JSON) for observability"
echo "  3. Run: make bench SCENARIO=mixed DURATION=300"
echo "  4. Validate: RAM P99 <9.5GB (target: 9.2GB)"
```

---

## 📊 Validation Checklist

Tras aplicar todos los patches:

- [ ] **Patch #1**: `python -c "from core.model_pool import ModelPool; pool = ModelPool({}); client = pool.get_skill_client('sql'); print('✅ OK')"`
- [ ] **Patch #2**: `docker-compose -f docker-compose.sentience.yml config` (sin errores)
- [ ] **Patch #3**: `pytest tests/test_entity_memory_skill_integration.py -v`
- [ ] **Patch #4**: Grafana dashboard muestra "Skill Hit Rate" panel
- [ ] **Benchmark**: `make bench SCENARIO=mixed` → RAM P99 <9.5GB
- [ ] **Health**: `docker ps | grep saraiskill` → todos "healthy"

---

**Autor**: Phoenix Integration Team  
**Versión**: 1.0 (Nov 8, 2025)  
**Roadmap**: ROADMAP_v2.15_SENTIENCE.md
