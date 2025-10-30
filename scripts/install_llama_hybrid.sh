#!/usr/bin/env bash
# scripts/install_llama_hybrid.sh
# Orquestador del sistema híbrido llama.cpp
# MANTRA: "Nunca falla, siempre fallback, un solo comando"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${HOME}/.cache/llama-hybrid"
STRATEGIES_JSON="${REPO_ROOT}/config/strategies.json"
LLAMA_BUILD_DIR="${REPO_ROOT}/.local"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# ============================================================================
# PASO 1: Detectar Hardware
# ============================================================================

detect_hardware() {
    log_info "Detectando hardware..."
    
    # Ejecutar script de detección
    if [[ ! -x "${REPO_ROOT}/scripts/detect_hardware.sh" ]]; then
        chmod +x "${REPO_ROOT}/scripts/detect_hardware.sh"
    fi
    
    DETECTION_OUTPUT=$("${REPO_ROOT}/scripts/detect_hardware.sh" 2>&1)
    
    # Parsear JSON de la primera línea (antes de los mensajes stderr)
    DETECTION_JSON=$(echo "$DETECTION_OUTPUT" | grep -m1 '^{' || echo "{}")
    
    # Extraer variables
    export STRATEGY=$(echo "$DETECTION_JSON" | grep -o '"strategy": "[^"]*"' | cut -d'"' -f4)
    export OPTIMAL_THREADS=$(echo "$DETECTION_JSON" | grep -o '"optimal_threads": [0-9]*' | awk '{print $2}')
    export ARCH_HINT=$(echo "$DETECTION_JSON" | grep -o '"arch_hint": "[^"]*"' | cut -d'"' -f4)
    export CPU_MODEL=$(echo "$DETECTION_JSON" | grep -o '"cpu_model": "[^"]*"' | cut -d'"' -f4)
    
    # Mostrar mensajes de detección (stderr del script)
    echo "$DETECTION_OUTPUT" | grep -v '^{' >&2
    
    log_info "CPU detectada: $CPU_MODEL"
    log_info "Estrategia seleccionada: $STRATEGY"
    log_info "Threads óptimos: $OPTIMAL_THREADS"
}

# ============================================================================
# PASO 2: Verificar Caché (evitar rebuild innecesario)
# ============================================================================

check_cache() {
    local strategy=$1
    local cache_path="${CACHE_DIR}/${strategy}"
    
    if [[ -d "$cache_path" ]] && [[ -f "${cache_path}/llama-cli" ]]; then
        log_info "Binario en caché encontrado: ${cache_path}/llama-cli"
        
        # Verificar que es ejecutable
        if [[ -x "${cache_path}/llama-cli" ]]; then
            log_success "Binario válido en caché (reutilizando)"
            return 0  # Caché válido
        fi
    fi
    
    return 1  # Caché no válido, build requerido
}

# ============================================================================
# PASO 3: Descargar Binario Pre-compilado (si disponible)
# ============================================================================

download_prebuilt() {
    local strategy=$1
    
    # Leer URL del JSON
    local url=$(grep -A20 "\"$strategy\"" "$STRATEGIES_JSON" | grep '"url"' | head -1 | cut -d'"' -f4)
    
    if [[ "$url" == "null" ]] || [[ -z "$url" ]]; then
        log_warning "No hay binario pre-compilado para $strategy (build requerido)"
        return 1
    fi
    
    log_info "Descargando binario pre-compilado..."
    log_info "URL: $url"
    
    mkdir -p "${CACHE_DIR}/${strategy}"
    
    # Descargar con curl (con retry)
    if curl -L -f --retry 3 --retry-delay 5 -o "${CACHE_DIR}/${strategy}/llama.zip" "$url"; then
        log_success "Descarga completa"
        
        # Extraer
        unzip -q "${CACHE_DIR}/${strategy}/llama.zip" -d "${CACHE_DIR}/${strategy}"
        
        # Buscar binario (puede estar en subdirectorio)
        find "${CACHE_DIR}/${strategy}" -name "llama-cli" -exec mv {} "${CACHE_DIR}/${strategy}/llama-cli" \;
        
        # Hacer ejecutable
        chmod +x "${CACHE_DIR}/${strategy}/llama-cli"
        
        log_success "Binario listo: ${CACHE_DIR}/${strategy}/llama-cli"
        return 0
    else
        log_error "Descarga falló (intentando build desde fuentes)"
        return 1
    fi
}

# ============================================================================
# PASO 4: Build desde Fuentes (OpenBLAS si requerido)
# ============================================================================

build_from_source() {
    local strategy=$1
    
    log_info "Iniciando build desde fuentes para estrategia: $strategy"
    
    # Leer configuración del JSON
    local openblas_required=$(grep -A20 "\"$strategy\"" "$STRATEGIES_JSON" | grep '"openblas_required"' | head -1 | grep -o 'true\|false')
    local cmake_flags=$(grep -A20 "\"$strategy\"" "$STRATEGIES_JSON" | grep '"cmake_flags"' | head -1 | cut -d'"' -f4)
    local build_time=$(grep -A20 "\"$strategy\"" "$STRATEGIES_JSON" | grep '"build_time_min"' | head -1 | grep -o '[0-9]*')
    
    log_info "Tiempo estimado: ~${build_time} minutos"
    log_info "Flags CMake: $cmake_flags"
    
    # PASO 4.1: OpenBLAS (si requerido)
    if [[ "$openblas_required" == "true" ]]; then
        log_info "OpenBLAS requerido para esta estrategia"
        
        if ! install_openblas; then
            log_error "OpenBLAS installation falló (fallback a generic)"
            export STRATEGY="generic"
            build_from_source "generic"
            return $?
        fi
    fi
    
    # PASO 4.2: Clonar llama.cpp (si no existe)
    if [[ ! -d "${LLAMA_BUILD_DIR}/llama.cpp" ]]; then
        log_info "Clonando llama.cpp..."
        git clone https://github.com/ggerganov/llama.cpp.git "${LLAMA_BUILD_DIR}/llama.cpp"
    else
        log_info "Actualizando llama.cpp..."
        cd "${LLAMA_BUILD_DIR}/llama.cpp"
        git pull origin master || log_warning "Git pull falló (usando versión existente)"
    fi
    
    # PASO 4.3: Build con CMake
    cd "${LLAMA_BUILD_DIR}/llama.cpp"
    mkdir -p build && cd build
    
    log_info "Ejecutando CMake..."
    
    # Construir comando CMake dinámicamente
    CMAKE_CMD="cmake .. $cmake_flags -DCMAKE_BUILD_TYPE=Release"
    
    # Añadir flags de arquitectura específica
    if [[ "$strategy" == "avx2-blas" ]]; then
        CMAKE_CMD="$CMAKE_CMD -DCMAKE_C_FLAGS=\"-march=native -mtune=${ARCH_HINT:-generic} -O3\""
    fi
    
    eval $CMAKE_CMD
    
    log_info "Compilando (usando $(nproc) cores)..."
    make -j$(nproc)
    
    # PASO 4.4: Copiar binario al caché
    mkdir -p "${CACHE_DIR}/${strategy}"
    cp llama-cli "${CACHE_DIR}/${strategy}/llama-cli"
    chmod +x "${CACHE_DIR}/${strategy}/llama-cli"
    
    log_success "Build completado: ${CACHE_DIR}/${strategy}/llama-cli"
    
    # PASO 4.5: Generar metadata
    generate_metadata "$strategy"
}

# ============================================================================
# PASO 4.1: Instalación de OpenBLAS
# ============================================================================

install_openblas() {
    log_info "Verificando OpenBLAS..."
    
    # Tier 1: pkg-config (sistema)
    if pkg-config --exists openblas; then
        log_success "OpenBLAS encontrado en sistema (pkg-config)"
        export OPENBLAS_LIB=$(pkg-config --libs openblas)
        return 0
    fi
    
    # Tier 2: Path común (/usr/lib)
    if [[ -f "/usr/lib/x86_64-linux-gnu/libopenblas.so" ]]; then
        log_success "OpenBLAS encontrado en /usr/lib"
        export OPENBLAS_LIB="-L/usr/lib/x86_64-linux-gnu -lopenblas"
        export PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH"
        return 0
    fi
    
    # Tier 3: Build desde fuentes
    log_warning "OpenBLAS no encontrado, compilando desde fuentes (~10 min)..."
    
    OPENBLAS_DIR="${LLAMA_BUILD_DIR}/OpenBLAS"
    
    if [[ ! -d "$OPENBLAS_DIR" ]]; then
        git clone https://github.com/xianyi/OpenBLAS.git "$OPENBLAS_DIR"
    fi
    
    cd "$OPENBLAS_DIR"
    
    # Build con target específico (Skylake, Haswell, etc.)
    local target=${ARCH_HINT^^}  # Uppercase
    
    log_info "Compilando OpenBLAS para target: $target"
    
    make -j$(nproc) \
        USE_OPENMP=1 \
        USE_THREAD=1 \
        NO_SHARED=0 \
        TARGET=${target:-CORE2} 2>&1 | tee openblas_build.log
    
    if [[ $? -ne 0 ]]; then
        log_error "OpenBLAS build falló"
        return 1
    fi
    
    make install PREFIX="${LLAMA_BUILD_DIR}/openblas"
    
    export OPENBLAS_LIB="-L${LLAMA_BUILD_DIR}/openblas/lib -lopenblas"
    export PKG_CONFIG_PATH="${LLAMA_BUILD_DIR}/openblas/lib/pkgconfig:$PKG_CONFIG_PATH"
    
    log_success "OpenBLAS compilado correctamente"
    return 0
}

# ============================================================================
# PASO 5: Generar Metadata (build_info.json)
# ============================================================================

generate_metadata() {
    local strategy=$1
    
    local metadata_file="${CACHE_DIR}/${strategy}/build_info.json"
    
    cat > "$metadata_file" << EOF
{
  "build_type": "$strategy",
  "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "cpu_model": "$CPU_MODEL",
  "optimal_threads": $OPTIMAL_THREADS,
  "arch_hint": "${ARCH_HINT:-generic}",
  "openblas_enabled": $(grep -q BLAS <<< "$strategy" && echo true || echo false),
  "expected_toks_solar_q4": "$(grep -A20 "\"$strategy\"" "$STRATEGIES_JSON" | grep '"expected_toks_solar_q4"' | cut -d'"' -f4)",
  "binary_path": "${CACHE_DIR}/${strategy}/llama-cli",
  "hash": "$(sha256sum "${CACHE_DIR}/${strategy}/llama-cli" | awk '{print $1}')"
}
EOF
    
    log_success "Metadata generada: $metadata_file"
}

# ============================================================================
# PASO 6: Symlink al binario activo (para python binding)
# ============================================================================

create_symlinks() {
    local strategy=$1
    
    log_info "Creando symlinks..."
    
    mkdir -p "${LLAMA_BUILD_DIR}/bin"
    
    ln -sf "${CACHE_DIR}/${strategy}/llama-cli" "${LLAMA_BUILD_DIR}/bin/llama-cli"
    ln -sf "${CACHE_DIR}/${strategy}/build_info.json" "${LLAMA_BUILD_DIR}/build_info.json"
    
    log_success "Binario activo: ${LLAMA_BUILD_DIR}/bin/llama-cli"
}

# ============================================================================
# PASO 7: Reinstalar llama-cpp-python (para usar nuevo binario)
# ============================================================================

reinstall_python_binding() {
    log_info "Reinstalando llama-cpp-python..."
    
    # Forzar reinstalación con el binario compilado
    CMAKE_ARGS="-DLLAMA_NATIVE=ON" \
    FORCE_CMAKE=1 \
    pip install --force-reinstall --no-cache-dir llama-cpp-python
    
    log_success "llama-cpp-python reinstalado"
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

main() {
    log_info "=== llama.cpp Hybrid Install System v1.0 ==="
    log_info "Mantra: Un comando, nunca falla, siempre fallback"
    echo ""
    
    # PASO 1: Detectar hardware
    detect_hardware
    echo ""
    
    # PASO 2: Verificar caché
    if check_cache "$STRATEGY"; then
        log_info "Reutilizando binario en caché (skip build)"
        create_symlinks "$STRATEGY"
        
        # Mostrar info
        cat "${CACHE_DIR}/${STRATEGY}/build_info.json"
        
        log_success "Instalación completa (caché hit)"
        exit 0
    fi
    echo ""
    
    # PASO 3: Intentar descarga de binario pre-compilado
    if download_prebuilt "$STRATEGY"; then
        log_info "Binario pre-compilado descargado correctamente"
        create_symlinks "$STRATEGY"
        generate_metadata "$STRATEGY"
        
        log_success "Instalación completa (binario pre-compilado)"
        exit 0
    fi
    echo ""
    
    # PASO 4: Build desde fuentes (con fallback a generic si falla)
    if ! build_from_source "$STRATEGY"; then
        log_error "Build falló para estrategia: $STRATEGY"
        
        if [[ "$STRATEGY" != "generic" ]]; then
            log_warning "Fallback a estrategia GENERIC..."
            export STRATEGY="generic"
            
            # Intentar descarga de generic
            if download_prebuilt "generic"; then
                create_symlinks "generic"
                generate_metadata "generic"
                log_success "Instalación completa (fallback a generic)"
                exit 0
            else
                log_error "Fallback también falló. Instalación abortada."
                exit 1
            fi
        else
            log_error "No se pudo completar la instalación"
            exit 1
        fi
    fi
    
    # PASO 5: Crear symlinks
    create_symlinks "$STRATEGY"
    
    # PASO 6: Reinstalar python binding (opcional)
    if [[ "${INSTALL_PYTHON_BINDING:-yes}" == "yes" ]]; then
        reinstall_python_binding
    fi
    
    echo ""
    log_success "=== Instalación completa ==="
    log_info "Estrategia: $STRATEGY"
    log_info "Threads óptimos: $OPTIMAL_THREADS"
    log_info "Binario: ${LLAMA_BUILD_DIR}/bin/llama-cli"
    
    # Mostrar metadata final
    cat "${CACHE_DIR}/${STRATEGY}/build_info.json"
}

# Ejecutar
main "$@"
