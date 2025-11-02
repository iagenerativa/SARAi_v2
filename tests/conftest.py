"""
Pytest configuration for SARAi test suite

FASE 5: Optimización - Test Infrastructure
Fecha: 2 Noviembre 2025

Configuración:
- Fixtures compartidos
- Markers personalizados
- Plugins (xdist, cov, profiling)
"""

import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== MARKERS ====================

def pytest_configure(config):
    """Configurar markers personalizados"""
    config.addinivalue_line(
        "markers", "slow: Marca tests lentos (>5s)"
    )
    config.addinivalue_line(
        "markers", "fast: Marca tests rápidos (<1s)"
    )
    config.addinivalue_line(
        "markers", "integration: Tests de integración E2E"
    )
    config.addinivalue_line(
        "markers", "unit: Tests unitarios aislados"
    )
    config.addinivalue_line(
        "markers", "security: Tests de seguridad (Safe Mode, chaos)"
    )
    config.addinivalue_line(
        "markers", "performance: Tests de performance (latencia, RAM)"
    )


# ==================== FIXTURES ====================

@pytest.fixture(scope="session")
def project_root():
    """Ruta raíz del proyecto"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def tests_dir():
    """Directorio de tests"""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def golden_queries_path(tests_dir):
    """Path a golden queries"""
    return tests_dir / "golden_queries.jsonl"


@pytest.fixture(scope="function")
def temp_state_dir(tmp_path):
    """Directorio temporal para estado de tests"""
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    return state_dir


@pytest.fixture(scope="function")
def temp_logs_dir(tmp_path):
    """Directorio temporal para logs de tests"""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


@pytest.fixture(scope="session")
def hmac_secret_key():
    """HMAC secret key para tests"""
    return os.getenv("HMAC_SECRET_KEY", "test-secret-key-do-not-use-in-production")


# ==================== HOOKS ====================

def pytest_collection_modifyitems(config, items):
    """
    Modificar items de tests durante collection
    
    - Auto-marcar tests lentos basado en nombre
    - Auto-marcar tests de integración
    """
    for item in items:
        # Auto-marcar tests lentos
        if "fast_lane" in item.nodeid or "chaos" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Auto-marcar tests rápidos
        if "unit" in item.nodeid or item.parent.name.startswith("test_unit_"):
            item.add_marker(pytest.mark.fast)
        
        # Auto-marcar tests de seguridad
        if "safe_mode" in item.nodeid or "chaos" in item.nodeid:
            item.add_marker(pytest.mark.security)
        
        # Auto-marcar tests de performance
        if "fast_lane" in item.nodeid or "regression" in item.nodeid:
            item.add_marker(pytest.mark.performance)


def pytest_report_header(config):
    """Header personalizado en reports"""
    return [
        "SARAi v2.14+ Test Suite",
        "FASE 5: Optimización - Parallel Testing & Coverage",
        f"Python: {sys.version.split()[0]}",
    ]


# ==================== PARAMETRIZE HELPERS ====================

# Configuraciones comunes para parametrize
CORRUPTION_TYPES = ["sha256", "hmac"]
REGRESSION_FACTORS = [0.70, 0.90, 1.0, 1.05]
PRIORITY_LEVELS = ["critical", "high", "normal", "low"]
