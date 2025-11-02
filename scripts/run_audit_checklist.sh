#!/usr/bin/env bash
###############################################################################
# SARAi v2.14 - Script de Auditoría Completa
###############################################################################
#
# Ejecuta las 15 secciones del AUDIT_CHECKLIST.md y genera un informe
# markdown en logs/audit_report_YYYY-MM-DD.md
#
# Uso:
#   bash scripts/run_audit_checklist.sh
#   bash scripts/run_audit_checklist.sh --verbose
#   bash scripts/run_audit_checklist.sh --section 1-8
#
###############################################################################

set -euo pipefail

# Colores ANSI
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

# Variables globales
VERBOSE=false
SECTION_FILTER=""
REPORT_FILE="logs/audit_report_$(date +%Y-%m-%d).md"
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# Crear directorio de logs si no existe
mkdir -p logs

###############################################################################
# Funciones de utilidad
###############################################################################

print_header() {
    echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}${BLUE}$1${RESET}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
}

print_result() {
    local status=$1
    local message=$2
    local details=${3:-}
    
    case $status in
        PASS)
            echo -e "${GREEN}✅ PASS${RESET} | $message"
            echo "✅ PASS | $message" >> "$REPORT_FILE"
            ((PASS_COUNT++))
            ;;
        FAIL)
            echo -e "${RED}❌ FAIL${RESET} | $message"
            echo "❌ FAIL | $message" >> "$REPORT_FILE"
            if [[ -n "$details" ]]; then
                echo -e "   ${YELLOW}→ $details${RESET}"
                echo "   → $details" >> "$REPORT_FILE"
            fi
            ((FAIL_COUNT++))
            ;;
        SKIP)
            echo -e "${YELLOW}⊘  SKIP${RESET} | $message"
            echo "⊘  SKIP | $message" >> "$REPORT_FILE"
            if [[ -n "$details" ]]; then
                echo -e "   ${YELLOW}→ $details${RESET}"
                echo "   → $details" >> "$REPORT_FILE"
            fi
            ((SKIP_COUNT++))
            ;;
    esac
}

run_command() {
    local cmd=$1
    local timeout=${2:-30}
    
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${BLUE}  $ $cmd${RESET}"
    fi
    
    timeout "$timeout" bash -c "$cmd" 2>&1
}

###############################################################################
# Sección 1: Configuración Base
###############################################################################

audit_section_1() {
    print_header "1. CONFIGURACIÓN BASE"
    echo -e "\n## 1. Configuración Base\n" >> "$REPORT_FILE"
    
    # 1.1 Variables de entorno críticas
    for var in OLLAMA_BASE_URL SOLAR_MODEL_NAME HOME_ASSISTANT_URL; do
        if [[ -n "${!var:-}" ]]; then
            print_result PASS "Variable $var definida: ${!var}"
        else
            print_result FAIL "Variable $var no definida" "Añadir a .env"
        fi
    done
    
    # 1.2 Modelos configurados
    if python3 -c "from core.unified_model_wrapper import ModelRegistry; r = ModelRegistry(); r.load_config(); exit(0 if len(r._config) >= 8 else 1)" 2>/dev/null; then
        model_count=$(python3 -c "from core.unified_model_wrapper import ModelRegistry; r = ModelRegistry(); r.load_config(); print(len(r._config))")
        print_result PASS "Modelos configurados: $model_count/8"
    else
        print_result FAIL "Menos de 8 modelos configurados" "Revisar config/models.yaml"
    fi
    
    # 1.3 Sin IPs hardcodeadas
    if ! grep -rq "192\.168\|10\.\|172\." config/ 2>/dev/null; then
        print_result PASS "Sin IPs hardcodeadas en config/"
    else
        ip_count=$(grep -r "192\.168\|10\.\|172\." config/ 2>/dev/null | wc -l)
        print_result FAIL "IPs hardcodeadas encontradas: $ip_count" "Usar variables de entorno"
    fi
}

###############################################################################
# Sección 2: Health Endpoints
###############################################################################

audit_section_2() {
    print_header "2. HEALTH ENDPOINTS"
    echo -e "\n## 2. Health Endpoints\n" >> "$REPORT_FILE"
    
    # 2.1 /health HTML
    if curl -sf http://localhost:8080/health 2>/dev/null | grep -q "HEALTHY"; then
        print_result PASS "/health endpoint (HTML)"
    else
        print_result SKIP "/health endpoint no responde" "Dashboard no está corriendo (OK)"
    fi
    
    # 2.2 /health JSON
    if curl -sf -H "Accept: application/json" http://localhost:8080/health 2>/dev/null | jq -e '.status' >/dev/null 2>&1; then
        print_result PASS "/health endpoint (JSON)"
    else
        print_result SKIP "/health JSON no responde" "Dashboard no está corriendo (OK)"
    fi
    
    # 2.3 /metrics Prometheus
    if curl -sf http://localhost:8080/metrics 2>/dev/null | grep -q "sarai_"; then
        print_result PASS "/metrics Prometheus"
    else
        print_result SKIP "/metrics no responde" "Dashboard no está corriendo (OK)"
    fi
}

###############################################################################
# Sección 3: Tests
###############################################################################

audit_section_3() {
    print_header "3. TESTS UNITARIOS E INTEGRACIÓN"
    echo -e "\n## 3. Tests Unitarios e Integración\n" >> "$REPORT_FILE"
    
    # 3.1 Tests del wrapper
    if [[ -f ".venv/bin/pytest" ]]; then
        if .venv/bin/pytest tests/test_unified_wrapper.py -v --tb=short -q 2>&1 | grep -q "passed"; then
            test_count=$(.venv/bin/pytest tests/test_unified_wrapper.py -v --tb=short -q 2>&1 | grep -oP '\d+(?= passed)' || echo "?")
            print_result PASS "Unified Wrapper tests: $test_count passed"
        else
            print_result FAIL "Unified Wrapper tests fallaron" "Ejecutar pytest con --tb=long"
        fi
    else
        print_result SKIP "pytest no encontrado" "Ejecutar 'make install' primero"
    fi
    
    # 3.2 Tests de integración
    if [[ -f ".venv/bin/pytest" ]] && [[ -f "tests/test_unified_wrapper_integration.py" ]]; then
        if .venv/bin/pytest tests/test_unified_wrapper_integration.py -v --tb=short -q 2>&1 | grep -q "passed"; then
            test_count=$(.venv/bin/pytest tests/test_unified_wrapper_integration.py -v --tb=short -q 2>&1 | grep -oP '\d+(?= passed)' || echo "?")
            print_result PASS "Tests de integración: $test_count passed"
        else
            print_result FAIL "Tests de integración fallaron"
        fi
    else
        print_result SKIP "Tests de integración no disponibles"
    fi
}

###############################################################################
# Sección 4: Auditoría de Logs
###############################################################################

audit_section_4() {
    print_header "4. AUDITORÍA DE LOGS (HMAC + SHA-256)"
    echo -e "\n## 4. Auditoría de Logs\n" >> "$REPORT_FILE"
    
    # 4.1 Logs estructurados
    if [[ -d "logs" ]]; then
        log_count=$(find logs -name "*.jsonl" 2>/dev/null | wc -l)
        if [[ $log_count -gt 0 ]]; then
            print_result PASS "Logs estructurados: $log_count archivos JSONL"
        else
            print_result SKIP "No hay logs JSONL" "Sistema recién instalado (OK)"
        fi
    else
        print_result SKIP "Directorio logs/ no existe" "Sistema recién instalado (OK)"
    fi
    
    # 4.2 Sidecars de verificación
    sidecar_count=$(find logs -name "*.sha256" -o -name "*.hmac" 2>/dev/null | wc -l)
    if [[ $sidecar_count -gt 0 ]]; then
        print_result PASS "Sidecars de verificación: $sidecar_count archivos"
    else
        print_result SKIP "No hay sidecars" "Sistema recién instalado (OK)"
    fi
}

###############################################################################
# Sección 5: Supply Chain (Cosign + SBOM)
###############################################################################

audit_section_5() {
    print_header "5. SUPPLY CHAIN (COSIGN + SBOM)"
    echo -e "\n## 5. Supply Chain\n" >> "$REPORT_FILE"
    
    # 5.1 Cosign instalado
    if command -v cosign >/dev/null 2>&1; then
        cosign_version=$(cosign version 2>&1 | head -n1 || echo "unknown")
        print_result PASS "Cosign instalado: $cosign_version"
    else
        print_result SKIP "Cosign no instalado" "Opcional para desarrollo local"
    fi
    
    # 5.2 Workflow de release
    if [[ -f ".github/workflows/release.yml" ]]; then
        print_result PASS "Workflow de release configurado"
    else
        print_result FAIL "Workflow de release no encontrado"
    fi
}

###############################################################################
# Sección 6: Docker Hardening
###############################################################################

audit_section_6() {
    print_header "6. HARDENING DE CONTENEDORES"
    echo -e "\n## 6. Hardening de Contenedores\n" >> "$REPORT_FILE"
    
    # Verificar si Docker está disponible
    if ! command -v docker >/dev/null 2>&1; then
        print_result SKIP "Docker no instalado" "Sección completa skip"
        return
    fi
    
    # Verificar si el contenedor existe
    if ! docker inspect sarai-omni-engine >/dev/null 2>&1; then
        print_result SKIP "Contenedor sarai-omni-engine no existe" "Ejecutar docker-compose up -d"
        return
    fi
    
    # 6.1 no-new-privileges
    if docker inspect sarai-omni-engine 2>/dev/null | jq -e '.[0].HostConfig.SecurityOpt[] | select(. == "no-new-privileges:true")' >/dev/null 2>&1; then
        print_result PASS "no-new-privileges activado"
    else
        print_result FAIL "no-new-privileges NO activado" "Añadir a docker-compose.override.yml"
    fi
    
    # 6.2 cap_drop ALL
    if docker inspect sarai-omni-engine 2>/dev/null | jq -e '.[0].HostConfig.CapDrop[] | select(. == "ALL")' >/dev/null 2>&1; then
        print_result PASS "cap_drop ALL activado"
    else
        print_result FAIL "cap_drop ALL NO activado"
    fi
    
    # 6.3 read-only filesystem
    if docker inspect sarai-omni-engine 2>/dev/null | jq -e '.[0].HostConfig.ReadonlyRootfs' | grep -q "true"; then
        print_result PASS "read-only filesystem activado"
    else
        print_result FAIL "read-only filesystem NO activado"
    fi
}

###############################################################################
# Sección 7: Skills Phoenix
###############################################################################

audit_section_7() {
    print_header "7. SKILLS PHOENIX (DETECCIÓN)"
    echo -e "\n## 7. Skills Phoenix\n" >> "$REPORT_FILE"
    
    # Verificar que core/mcp.py existe
    if [[ ! -f "core/mcp.py" ]]; then
        print_result SKIP "core/mcp.py no encontrado"
        return
    fi
    
    # Test de detección de skills
    test_cases=(
        "Cómo crear una función en Python:programming"
        "Analizar error de base de datos:diagnosis"
        "Estrategia de inversión ROI:financial"
        "Escribe una historia corta:creative"
    )
    
    pass_count=0
    for test_case in "${test_cases[@]}"; do
        query="${test_case%%:*}"
        expected="${test_case##*:}"
        
        detected=$(python3 -c "
from core.mcp import detect_and_apply_skill
skill = detect_and_apply_skill('$query', 'solar')
print(skill['name'] if skill else 'none')
" 2>/dev/null || echo "error")
        
        if [[ "$detected" == "$expected" ]]; then
            print_result PASS "\"${query:0:30}...\" → $detected"
            ((pass_count++))
        else
            print_result FAIL "\"${query:0:30}...\" → $detected (esperado: $expected)"
        fi
    done
    
    if [[ $pass_count -eq ${#test_cases[@]} ]]; then
        echo -e "\n${GREEN}✅ Skills Phoenix: $pass_count/${#test_cases[@]} tests passing${RESET}"
    fi
}

###############################################################################
# Sección 8: Layers Architecture
###############################################################################

audit_section_8() {
    print_header "8. LAYERS ARCHITECTURE"
    echo -e "\n## 8. Layers Architecture\n" >> "$REPORT_FILE"
    
    # 8.1 Layer 2: Tone memory
    if [[ -f "state/layer2_tone_memory.jsonl" ]]; then
        entries=$(wc -l < state/layer2_tone_memory.jsonl)
        if [[ $entries -le 256 ]]; then
            print_result PASS "Tone memory: $entries entradas (max 256)"
        else
            print_result FAIL "Tone memory excede límite: $entries > 256"
        fi
    else
        print_result SKIP "Tone memory no inicializado" "Sistema recién instalado (OK)"
    fi
    
    # 8.2 Layer 3: Tone bridge
    if python3 -c "from core.layer3_fluidity.tone_bridge import get_tone_bridge; bridge = get_tone_bridge(); profile = bridge.update('happy', 0.8, 0.7); exit(0 if profile.style == 'energetic_positive' else 1)" 2>/dev/null; then
        print_result PASS "Tone bridge: estilo inferido correctamente"
    else
        print_result FAIL "Tone bridge: error en inferencia de estilo"
    fi
}

###############################################################################
# Sección 9: Extended - models.yaml
###############################################################################

audit_section_9() {
    print_header "9. EXTENDED: MODELS.YAML"
    echo -e "\n## 9. Extended: models.yaml\n" >> "$REPORT_FILE"
    
    # 9.1 Archivo existe
    if [[ -f "config/models.yaml" ]]; then
        print_result PASS "config/models.yaml existe"
    else
        print_result FAIL "config/models.yaml no encontrado"
        return
    fi
    
    # 9.2 YAML válido
    if python3 -c "import yaml; yaml.safe_load(open('config/models.yaml'))" 2>/dev/null; then
        print_result PASS "YAML sintácticamente correcto"
    else
        print_result FAIL "YAML con errores de sintaxis"
    fi
    
    # 9.3 Modelos mínimos configurados
    model_count=$(python3 -c "import yaml; print(len(yaml.safe_load(open('config/models.yaml'))))" 2>/dev/null || echo "0")
    if [[ $model_count -ge 8 ]]; then
        print_result PASS "Modelos configurados: $model_count/8"
    else
        print_result FAIL "Modelos insuficientes: $model_count < 8"
    fi
}

###############################################################################
# Sección 10: Extended - Memory Limits
###############################################################################

audit_section_10() {
    print_header "10. EXTENDED: MEMORY LIMITS"
    echo -e "\n## 10. Extended: Memory Limits\n" >> "$REPORT_FILE"
    
    # 10.1 RAM total disponible
    if command -v free >/dev/null 2>&1; then
        total_gb=$(free -g | awk '/^Mem:/{print $2}')
        if [[ $total_gb -ge 12 ]]; then
            print_result PASS "RAM total: ${total_gb}GB (≥12GB)"
        else
            print_result FAIL "RAM insuficiente: ${total_gb}GB < 12GB"
        fi
    else
        print_result SKIP "Comando 'free' no disponible"
    fi
    
    # 10.2 Límite max_concurrent_llms
    if grep -q "max_concurrent_llms.*2" config/sarai.yaml 2>/dev/null; then
        print_result PASS "max_concurrent_llms: 2 (correcto)"
    else
        print_result FAIL "max_concurrent_llms no configurado correctamente"
    fi
}

###############################################################################
# Sección 11: Extended - Security (Firejail/chattr)
###############################################################################

audit_section_11() {
    print_header "11. EXTENDED: SECURITY (FIREJAIL/CHATTR)"
    echo -e "\n## 11. Extended: Security\n" >> "$REPORT_FILE"
    
    # 11.1 Firejail instalado
    if command -v firejail >/dev/null 2>&1; then
        firejail_version=$(firejail --version 2>&1 | head -n1 || echo "unknown")
        print_result PASS "Firejail instalado: $firejail_version"
    else
        print_result SKIP "Firejail no instalado" "Opcional para sandboxing de skills"
    fi
    
    # 11.2 chattr disponible
    if command -v chattr >/dev/null 2>&1; then
        print_result PASS "chattr disponible (para logs append-only)"
    else
        print_result SKIP "chattr no disponible"
    fi
}

###############################################################################
# Sección 12: Extended - Networking
###############################################################################

audit_section_12() {
    print_header "12. EXTENDED: NETWORKING"
    echo -e "\n## 12. Extended: Networking\n" >> "$REPORT_FILE"
    
    # 12.1 Ollama alcanzable
    ollama_url="${OLLAMA_BASE_URL:-http://localhost:11434}"
    if curl -sf "$ollama_url/api/version" >/dev/null 2>&1; then
        print_result PASS "Ollama alcanzable en $ollama_url"
    else
        print_result SKIP "Ollama no responde en $ollama_url" "Puede estar apagado (OK)"
    fi
    
    # 12.2 Home Assistant alcanzable
    if [[ -n "${HOME_ASSISTANT_URL:-}" ]]; then
        if curl -sf "$HOME_ASSISTANT_URL/api/" >/dev/null 2>&1; then
            print_result PASS "Home Assistant alcanzable"
        else
            print_result SKIP "Home Assistant no responde" "Puede estar apagado (OK)"
        fi
    else
        print_result SKIP "HOME_ASSISTANT_URL no configurada"
    fi
}

###############################################################################
# Sección 13: Extended - Fallbacks
###############################################################################

audit_section_13() {
    print_header "13. EXTENDED: FALLBACKS"
    echo -e "\n## 13. Extended: Fallbacks\n" >> "$REPORT_FILE"
    
    # 13.1 Safe Mode implementado
    if grep -q "is_safe_mode" core/audit.py 2>/dev/null; then
        print_result PASS "Safe Mode implementado en core/audit.py"
    else
        print_result FAIL "Safe Mode no encontrado"
    fi
    
    # 13.2 Fallback cascada en ModelPool
    if grep -q "fallback" core/model_pool.py 2>/dev/null; then
        print_result PASS "Fallback cascade en ModelPool"
    else
        print_result SKIP "Fallback cascade no implementado aún"
    fi
    
    # 13.3 Sentinel responses
    if grep -q "SENTINEL_RESPONSES" agents/rag_agent.py 2>/dev/null; then
        print_result PASS "Sentinel responses en RAG agent"
    else
        print_result SKIP "Sentinel responses no implementadas"
    fi
    
    # 13.4 Degradación elegante
    if grep -q "degrad" README.md 2>/dev/null; then
        print_result PASS "Degradación elegante documentada"
    else
        print_result SKIP "Degradación elegante no documentada"
    fi
}

###############################################################################
# Sección 14: Extended - E2E Latency
###############################################################################

audit_section_14() {
    print_header "14. EXTENDED: E2E LATENCY"
    echo -e "\n## 14. Extended: E2E Latency\n" >> "$REPORT_FILE"
    
    # 14.1 Benchmark script existe
    if [[ -f "scripts/benchmark_wrapper_overhead.py" ]]; then
        print_result PASS "Script de benchmark existe"
    else
        print_result FAIL "scripts/benchmark_wrapper_overhead.py no encontrado"
    fi
    
    # 14.2 Histórico de benchmarks
    if [[ -d "benchmarks/history" ]] && [[ $(ls -1 benchmarks/history/*.json 2>/dev/null | wc -l) -gt 0 ]]; then
        bench_count=$(ls -1 benchmarks/history/*.json 2>/dev/null | wc -l)
        print_result PASS "Histórico de benchmarks: $bench_count archivos"
    else
        print_result SKIP "Sin histórico de benchmarks" "Ejecutar 'make benchmark VERSION=vX.X'"
    fi
    
    # 14.3 KPIs de latencia definidos
    if grep -q "Latencia P50" README.md 2>/dev/null; then
        print_result PASS "KPIs de latencia documentados en README"
    else
        print_result FAIL "KPIs de latencia no documentados"
    fi
}

###############################################################################
# Sección 15: Extended - KPI Matrix
###############################################################################

audit_section_15() {
    print_header "15. EXTENDED: KPI MATRIX"
    echo -e "\n## 15. Extended: KPI Matrix\n" >> "$REPORT_FILE"
    
    # 15.1 Tabla de KPIs en README
    if grep -q "| KPI |" README.md 2>/dev/null; then
        print_result PASS "Tabla de KPIs presente en README"
    else
        print_result FAIL "Tabla de KPIs no encontrada en README"
    fi
    
    # 15.2 KPIs v2.14 documentados
    if grep -q "v2.14" README.md 2>/dev/null && grep -q "RAM P99" README.md 2>/dev/null; then
        print_result PASS "KPIs v2.14 documentados"
    else
        print_result FAIL "KPIs v2.14 no documentados"
    fi
    
    # 15.3 Status actual documentado
    if [[ -f "STATUS_ACTUAL.md" ]]; then
        if grep -q "v2.14" STATUS_ACTUAL.md 2>/dev/null; then
            print_result PASS "STATUS_ACTUAL.md actualizado a v2.14"
        else
            print_result FAIL "STATUS_ACTUAL.md desactualizado"
        fi
    else
        print_result FAIL "STATUS_ACTUAL.md no encontrado"
    fi
}

###############################################################################
# Main
###############################################################################

main() {
    # Parse argumentos
    while [[ $# -gt 0 ]]; do
        case $1 in
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --section|-s)
                SECTION_FILTER="$2"
                shift 2
                ;;
            *)
                echo "Uso: $0 [--verbose] [--section 1-5]"
                exit 1
                ;;
        esac
    done
    
    # Header del informe
    cat > "$REPORT_FILE" << EOF
# SARAi v2.14 - Informe de Auditoría

**Fecha**: $(date +"%Y-%m-%d %H:%M:%S")  
**Host**: $(hostname)  
**Usuario**: $(whoami)  
**Python**: $(python3 --version 2>&1)

---

EOF
    
    echo -e "${BOLD}${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║       SARAi v2.14 - Auditoría Completa del Sistema              ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}\n"
    
    # Ejecutar secciones
    if [[ -z "$SECTION_FILTER" ]]; then
        # Todas las secciones (1-15)
        audit_section_1
        audit_section_2
        audit_section_3
        audit_section_4
        audit_section_5
        audit_section_6
        audit_section_7
        audit_section_8
        audit_section_9
        audit_section_10
        audit_section_11
        audit_section_12
        audit_section_13
        audit_section_14
        audit_section_15
    else
        # Secciones específicas
        IFS='-' read -ra RANGE <<< "$SECTION_FILTER"
        start=${RANGE[0]:-1}
        end=${RANGE[1]:-$start}
        
        for i in $(seq "$start" "$end"); do
            "audit_section_$i"
        done
    fi
    
    # Resumen final
    print_header "RESUMEN"
    
    total=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
    pass_percent=$(( total > 0 ? PASS_COUNT * 100 / total : 0 ))
    
    echo -e "\n## Resumen Final\n" >> "$REPORT_FILE"
    echo "| Estado | Cantidad | Porcentaje |" >> "$REPORT_FILE"
    echo "|--------|----------|------------|" >> "$REPORT_FILE"
    echo "| ✅ PASS | $PASS_COUNT | ${pass_percent}% |" >> "$REPORT_FILE"
    echo "| ❌ FAIL | $FAIL_COUNT | $(( total > 0 ? FAIL_COUNT * 100 / total : 0 ))% |" >> "$REPORT_FILE"
    echo "| ⊘  SKIP | $SKIP_COUNT | $(( total > 0 ? SKIP_COUNT * 100 / total : 0 ))% |" >> "$REPORT_FILE"
    echo "| **TOTAL** | **$total** | **100%** |" >> "$REPORT_FILE"
    
    echo -e "\nTotal: ${GREEN}$PASS_COUNT PASS${RESET} | ${RED}$FAIL_COUNT FAIL${RESET} | ${YELLOW}$SKIP_COUNT SKIP${RESET}"
    echo -e "Porcentaje de aprobación: ${BOLD}${pass_percent}%${RESET}\n"
    
    if [[ $pass_percent -ge 95 ]]; then
        echo -e "${GREEN}${BOLD}✅ AUDITORÍA APROBADA${RESET}\n"
        echo -e "\n---\n\n**Resultado**: ✅ **AUDITORÍA APROBADA** (≥95%)" >> "$REPORT_FILE"
        exit_code=0
    elif [[ $pass_percent -ge 80 ]]; then
        echo -e "${YELLOW}${BOLD}⚠️  AUDITORÍA APROBADA CON OBSERVACIONES${RESET}\n"
        echo -e "\n---\n\n**Resultado**: ⚠️  **AUDITORÍA APROBADA CON OBSERVACIONES** (80-94%)" >> "$REPORT_FILE"
        exit_code=0
    else
        echo -e "${RED}${BOLD}❌ AUDITORÍA RECHAZADA${RESET}\n"
        echo -e "\n---\n\n**Resultado**: ❌ **AUDITORÍA RECHAZADA** (<80%)" >> "$REPORT_FILE"
        exit_code=1
    fi
    
    echo -e "Informe generado en: ${BLUE}$REPORT_FILE${RESET}\n"
    
    exit $exit_code
}

main "$@"
