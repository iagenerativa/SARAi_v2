@echo off
REM Pre-requisitos Check para Cuantizacion INT8
REM Ejecutar ANTES de quantize_windows.bat

echo ========================================
echo SARAi v2.16.1 - Verificacion Pre-requisitos
echo ========================================
echo.

set ERRORS=0

REM ============================================
REM 1. Python instalado y version correcta
REM ============================================
echo [1/7] Verificando Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   X Python NO encontrado en PATH
    echo   Instalar Python 3.10+ desde python.org
    set ERRORS=1
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo   OK Python encontrado: %PYTHON_VERSION%
)
echo.

REM ============================================
REM 2. RAM disponible (minimo 24GB)
REM ============================================
echo [2/7] Verificando RAM...
for /f "tokens=2 delims=:" %%i in ('wmic ComputerSystem get TotalPhysicalMemory /value') do set RAM_BYTES=%%i
set /a RAM_GB=%RAM_BYTES:~0,-9%
if %RAM_GB% LSS 24 (
    echo   ! ADVERTENCIA: RAM total: %RAM_GB%GB
    echo   Recomendado: 32GB minimo
    echo   Puede causar OOM durante cuantizacion
) else (
    echo   OK RAM total: %RAM_GB%GB
)
echo.

REM ============================================
REM 3. Espacio en disco (minimo 6GB)
REM ============================================
echo [3/7] Verificando espacio en disco...
for /f "tokens=3" %%i in ('dir /-c ^| findstr "bytes free"') do set DISK_FREE=%%i
set DISK_FREE=%DISK_FREE:,=%
set /a DISK_GB=%DISK_FREE:~0,-9%
if %DISK_GB% LSS 6 (
    echo   X Espacio insuficiente: %DISK_GB%GB
    echo   Requerido: 6GB minimo
    set ERRORS=1
) else (
    echo   OK Espacio libre: %DISK_GB%GB
)
echo.

REM ============================================
REM 4. Modelo FP32 presente
REM ============================================
echo [4/7] Verificando modelo FP32...
if not exist "models\onnx\agi_audio_core.onnx" (
    echo   X Modelo NO encontrado: models\onnx\agi_audio_core.onnx
    echo   Copiar modelo ONNX a models\onnx\ primero
    set ERRORS=1
) else (
    echo   OK Metadata encontrado: agi_audio_core.onnx
)

if not exist "models\onnx\agi_audio_core.onnx.data" (
    echo   X Pesos NO encontrados: models\onnx\agi_audio_core.onnx.data
    echo   Copiar .data a models\onnx\ primero
    set ERRORS=1
) else (
    for %%A in ("models\onnx\agi_audio_core.onnx.data") do set SIZE=%%~zA
    set /a SIZE_GB=%SIZE:~0,-9%
    if %SIZE_GB% LSS 4 (
        echo   X Pesos muy pequenos: %SIZE_GB%GB
        echo   Esperado: ~4.3GB
        echo   Modelo puede estar corrupto
        set ERRORS=1
    ) else (
        echo   OK Pesos encontrados: %SIZE_GB%GB
    )
)
echo.

REM ============================================
REM 5. GPU (opcional)
REM ============================================
echo [5/7] Verificando GPU (opcional)...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo   - GPU NVIDIA no detectada
    echo   Cuantizacion usara CPU (5-10 min)
    echo   Con GPU seria 2-3 min
) else (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader') do set GPU_NAME=%%i
    echo   OK GPU detectada: %GPU_NAME%
    echo   Cuantizacion usara GPU (2-3 min)
)
echo.

REM ============================================
REM 6. SSH a Linux (opcional ahora)
REM ============================================
echo [6/7] Verificando SSH a Linux (opcional)...
ssh -o ConnectTimeout=2 noel@agi1 "echo OK" >nul 2>&1
if %errorlevel% neq 0 (
    echo   - SSH no disponible (agi1)
    echo   Se necesitara para transferir modelo despues
    echo   Alternativa: Usar WinSCP manualmente
) else (
    echo   OK SSH a agi1 funcional
)
echo.

REM ============================================
REM 7. Dependencias Python (onnx, onnxruntime)
REM ============================================
echo [7/7] Verificando dependencias Python...
python -c "import onnx" >nul 2>&1
if %errorlevel% neq 0 (
    echo   - onnx NO instalado
    echo   Se instalara automaticamente
) else (
    for /f "tokens=2" %%i in ('python -c "import onnx; print(onnx.__version__)"') do set ONNX_VER=%%i
    echo   OK onnx instalado
)

python -c "import onnxruntime" >nul 2>&1
if %errorlevel% neq 0 (
    echo   - onnxruntime NO instalado
    echo   Se instalara automaticamente
) else (
    for /f "tokens=*" %%i in ('python -c "import onnxruntime; print(onnxruntime.__version__)"') do set ORT_VER=%%i
    echo   OK onnxruntime instalado: %ORT_VER%
    
    REM Verificar si tiene GPU support
    python -c "import onnxruntime as ort; 'CUDAExecutionProvider' in ort.get_available_providers() and exit(0) or exit(1)" >nul 2>&1
    if %errorlevel% equ 0 (
        echo   OK onnxruntime con soporte GPU
    ) else (
        echo   - onnxruntime SIN soporte GPU
        echo   Para GPU: pip install onnxruntime-gpu
    )
)
echo.

REM ============================================
REM RESUMEN
REM ============================================
echo ========================================
echo RESUMEN
echo ========================================
echo.

if %ERRORS% neq 0 (
    echo   X ERRORES CRITICOS encontrados
    echo.
    echo   Resolver los problemas marcados con X
    echo   antes de ejecutar quantize_windows.bat
    echo.
    pause
    exit /b 1
) else (
    echo   OK Todos los pre-requisitos cumplidos
    echo.
    echo ========================================
    echo SIGUIENTE PASO:
    echo ========================================
    echo.
    echo   Ejecutar cuantizacion:
    echo.
    echo      scripts\quantize_windows.bat
    echo.
    echo   Tiempo estimado:
    if exist "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" (
        echo      - 2-3 min (con GPU)
    ) else (
        echo      - 5-10 min (con CPU)
    )
    echo.
    echo ========================================
    pause
    exit /b 0
)
