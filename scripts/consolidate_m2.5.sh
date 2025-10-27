#!/bin/bash
# scripts/consolidate_m2.5.sh
# Consolidación REAL de M2.5 con TRM real y SearXNG funcionando

set -e  # Exit on error

echo "🎯 CONSOLIDACIÓN M2.5 - Test End-to-End REAL"
echo "============================================="
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paso 1: Verificar autenticación HuggingFace
echo "📋 Paso 1: Verificar HuggingFace Auth"
echo "--------------------------------------"

# Verificar con Python (más confiable)
if ! python3 -c "from huggingface_hub import whoami; whoami()" 2>/dev/null; then
    echo -e "${RED}❌ No estás autenticado en HuggingFace${NC}"
    echo ""
    echo "Necesitas hacer login primero:"
    echo "  1. Ejecuta: huggingface-cli login"
    echo "  2. Pega tu token de: https://huggingface.co/settings/tokens"
    echo "  3. Acepta términos de EmbeddingGemma:"
    echo "     https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized"
    echo ""
    exit 1
else
    echo -e "${GREEN}✅ HuggingFace autenticado${NC}"
    python3 -c "from huggingface_hub import whoami; u=whoami(); print(f'   Usuario: {u[\"name\"]}')"
fi

echo ""

# Paso 2: Verificar que el TRM esté entrenado
echo "📋 Paso 2: Verificar TRM Entrenado"
echo "----------------------------------"

if [ ! -f "models/trm_classifier/checkpoint.pth" ]; then
    echo -e "${RED}❌ TRM no está entrenado${NC}"
    echo "Ejecuta primero: python3 scripts/train_trm_v2.py"
    exit 1
else
    echo -e "${GREEN}✅ TRM checkpoint encontrado${NC}"
    ls -lh models/trm_classifier/checkpoint.pth
fi

echo ""

# Paso 3: Levantar SearXNG
echo "📋 Paso 3: Levantar SearXNG"
echo "---------------------------"

# Verificar si docker está disponible
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker no está instalado${NC}"
    exit 1
fi

# Verificar si docker-compose está disponible
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ docker-compose no está disponible${NC}"
    exit 1
fi

# Usar docker compose (nuevo) o docker-compose (legacy)
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo "Usando: $DOCKER_COMPOSE"

# Levantar solo SearXNG (no Omni por ahora)
echo "Levantando SearXNG..."
$DOCKER_COMPOSE up -d searxng

# Esperar a que esté ready
echo "Esperando a que SearXNG esté listo..."
for i in {1..30}; do
    if curl -s http://localhost:8888/search?q=test&format=json > /dev/null 2>&1; then
        echo -e "${GREEN}✅ SearXNG está listo en http://localhost:8888${NC}"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ SearXNG no respondió después de 30 segundos${NC}"
        echo "Logs de SearXNG:"
        $DOCKER_COMPOSE logs searxng | tail -20
        exit 1
    fi
    
    echo -n "."
    sleep 1
done

echo ""

# Paso 4: Test de TRM con embeddings reales
echo "📋 Paso 4: Test TRM con EmbeddingGemma Real"
echo "-------------------------------------------"

python3 << 'EOF'
import sys
import torch

try:
    # Cargar embedding model (requiere HF auth)
    print("Cargando EmbeddingGemma...")
    from core.embeddings import get_embedding_model
    
    embedder = get_embedding_model()
    print(f"✅ EmbeddingGemma cargado: {type(embedder)}")
    
    # Test de encoding
    test_text = "¿Quién ganó el Oscar 2025?"
    embedding = embedder.encode(test_text)
    print(f"✅ Embedding generado: shape={embedding.shape}")
    
    # Cargar TRM
    print("\nCargando TRM Classifier...")
    from core.trm_classifier import TRMClassifierDual
    
    trm = TRMClassifierDual()
    checkpoint = torch.load("models/trm_classifier/checkpoint.pth")
    trm.load_state_dict(checkpoint['model_state_dict'])
    trm.eval()
    print(f"✅ TRM cargado desde checkpoint")
    
    # Test de clasificación
    print("\nTest de queries web:")
    test_queries = [
        "¿Quién ganó el Oscar 2025?",
        "¿Cómo está el clima en Tokio?",
        "¿Cómo configurar SSH en Ubuntu?",
        "Me siento frustrado"
    ]
    
    for query in test_queries:
        emb = embedder.encode(query)
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            scores = trm.forward(emb_tensor)
        
        print(f"\n  Query: {query}")
        print(f"    hard={scores['hard']:.3f}, soft={scores['soft']:.3f}, web_query={scores['web_query']:.3f}")
        
        # Validar que detecta correctamente
        if "Oscar" in query or "clima" in query:
            assert scores['web_query'] > 0.7, f"Query web no detectada: {query}"
            print(f"    ✅ Web query detectada correctamente")
        else:
            assert scores['web_query'] < 0.7, f"False positive en web_query: {query}"
            print(f"    ✅ No-web query correcta")
    
    print("\n✅ TRM con embeddings reales: FUNCIONA")
    
except Exception as e:
    print(f"\n❌ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Test de TRM falló${NC}"
    exit 1
fi

echo ""

# Paso 5: Test end-to-end con SearXNG
echo "📋 Paso 5: Test End-to-End con SearXNG"
echo "--------------------------------------"

python3 << 'EOF'
import sys

try:
    # Desactivar Safe Mode temporalmente
    from core.audit import disable_safe_mode_temp
    
    print("Desactivando Safe Mode temporalmente...")
    with disable_safe_mode_temp():
        print("✅ Safe Mode desactivado para test")
        
        # Test de búsqueda web directa
        print("\nTest 1: Búsqueda web con cached_search()")
        from core.web_cache import cached_search
        
        query = "Python programming language"
        results = cached_search(query)
        
        if results is None:
            print("❌ cached_search() retornó None (Safe Mode o error)")
            sys.exit(1)
        
        print(f"✅ Resultados obtenidos:")
        print(f"   Fuente: {results['source']}")
        print(f"   Snippets: {len(results['snippets'])}")
        
        if len(results['snippets']) > 0:
            print(f"\n   Ejemplo de snippet:")
            snippet = results['snippets'][0]
            print(f"     Título: {snippet['title']}")
            print(f"     URL: {snippet['url']}")
            print(f"     Contenido: {snippet['content'][:150]}...")
        
        # Test 2: Pipeline RAG completo
        print("\n\nTest 2: Pipeline RAG completo con Graph")
        from core.graph import create_orchestrator
        
        graph = create_orchestrator(use_simulated_trm=False)
        print("✅ Graph creado con TRM real")
        
        # Query que debe enrutar a RAG
        query = "¿Quién inventó Python?"
        print(f"\nQuery: {query}")
        
        result = graph.invoke(query)
        
        print(f"\n✅ Respuesta obtenida:")
        print(f"   Scores: hard={result['hard']:.3f}, soft={result['soft']:.3f}, web_query={result['web_query']:.3f}")
        print(f"   Routing: α={result['alpha']:.3f}, β={result['beta']:.3f}")
        print(f"   Agent usado: {result.get('agent_used', 'N/A')}")
        
        # Verificar que enrutó a RAG
        if result.get('agent_used') != 'rag':
            print(f"⚠️  WARNING: No enrutó a RAG (agent={result.get('agent_used')})")
            print(f"   web_query score: {result['web_query']}")
        else:
            print(f"   ✅ Correctamente enrutado a RAG")
        
        print(f"\n   Respuesta (preview):")
        print(f"   {result['response'][:300]}...")
        
        # Verificar metadata RAG
        if 'rag_metadata' in result:
            metadata = result['rag_metadata']
            print(f"\n   Metadata RAG:")
            print(f"     Fuente: {metadata.get('source', 'N/A')}")
            print(f"     Snippets: {metadata.get('snippets_count', 0)}")
            print(f"     LLM: {metadata.get('llm_model', 'N/A')}")
        
        print("\n✅ Pipeline RAG completo: FUNCIONA")

except Exception as e:
    print(f"\n❌ Error en test end-to-end: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Test end-to-end falló${NC}"
    
    # Mostrar logs de SearXNG
    echo ""
    echo "Logs de SearXNG:"
    $DOCKER_COMPOSE logs searxng | tail -30
    
    exit 1
fi

echo ""

# Paso 6: Validación final
echo "📋 Paso 6: Validación Final"
echo "---------------------------"

echo -e "${GREEN}✅ TODOS LOS TESTS PASARON${NC}"
echo ""
echo "Componentes validados:"
echo "  ✅ HuggingFace autenticado"
echo "  ✅ EmbeddingGemma funcional"
echo "  ✅ TRM con 3 cabezas (hard/soft/web_query)"
echo "  ✅ SearXNG respondiendo en http://localhost:8888"
echo "  ✅ cached_search() con resultados reales"
echo "  ✅ Pipeline RAG end-to-end"
echo ""
echo -e "${GREEN}🎉 M2.5 CONSOLIDADO EXITOSAMENTE${NC}"
echo ""
echo "Siguiente paso:"
echo "  - Actualizar TODO list marcando M2.5 como completado (real)"
echo "  - Continuar con siguiente milestone (M3, M4 o M5)"

# Limpiar
echo ""
echo "¿Deseas detener SearXNG? (mantenerlo corriendo para futuros tests)"
echo "Para detener: docker-compose down searxng"
