@echo off
REM Cuantización INT8 para Windows
REM Ejecutar desde: SARAi_v2\

echo ========================================
echo SARAi v2.16.1 - Cuantizacion INT8
echo Windows 11 - GPU Accelerated
echo ========================================
echo.

REM Verificar que estamos en el directorio correcto
if not exist "models\onnx\agi_audio_core.onnx" (
    echo ERROR: Modelo no encontrado
    echo Asegurate de estar en el directorio raiz de SARAi_v2
    echo y que agi_audio_core.onnx este en models\onnx\
    pause
    exit /b 1
)

REM Verificar Python instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python no encontrado en PATH
    echo Instalar Python 3.10+ desde python.org
    pause
    exit /b 1
)

echo [1/3] Verificando dependencias...
python -c "import onnx" >nul 2>&1
if %errorlevel% neq 0 (
    echo Instalando onnx...
    pip install onnx
)

python -c "import onnxruntime" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Instalando onnxruntime-gpu para soporte GPU...
    echo (Si no tienes GPU, usa: pip install onnxruntime)
    pip install onnxruntime-gpu
)

echo.
echo [2/3] Ejecutando cuantizacion INT8...
echo Tiempo estimado: 2-3 minutos con GPU
echo.

python scripts\quantize_onnx_int8_windows.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Cuantizacion fallo
    pause
    exit /b 1
)

echo.
echo [3/3] Cuantizacion completada exitosamente!
echo.
echo Archivos generados:
dir models\onnx\agi_audio_core_int8.onnx*
echo.
echo ========================================
echo SIGUIENTE PASO:
echo ========================================
echo.
echo Transferir modelo cuantizado a Linux:
echo.
echo   1. Abrir PowerShell/CMD
echo   2. Ejecutar:
echo.
echo      scp models\onnx\agi_audio_core_int8.onnx noel@agi1:~/SARAi_v2/models/onnx/
echo      scp models\onnx\agi_audio_core_int8.onnx.data noel@agi1:~/SARAi_v2/models/onnx/
echo.
echo   3. En Linux, actualizar config/sarai.yaml
echo.
echo ========================================
pause
