"""
Tests para Vision Agent v2.12 (Qwen3-VL-4B)
Validación de análisis de imágenes y gestión de memoria
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from agents.vision_agent import VisionAgent, create_vision_agent


class TestVisionAgentBasics:
    """Tests básicos de funcionalidad"""
    
    @pytest.fixture
    def mock_model_pool(self):
        """Mock de ModelPool"""
        pool = MagicMock()
        
        # Mock de modelo Qwen3-VL
        vision_model = MagicMock()
        vision_model.create_completion = lambda prompt, **kwargs: {
            "choices": [{
                "text": "Esta imagen muestra un diagrama de arquitectura con 3 componentes principales."
            }]
        }
        
        pool.get.return_value = vision_model
        return pool
    
    def test_analyze_image_from_path(self, mock_model_pool, tmp_path):
        """Test V1: Analizar imagen desde ruta"""
        # Crear imagen fake
        fake_image = tmp_path / "test.png"
        fake_image.write_bytes(b"fake_png_data")
        
        agent = VisionAgent(mock_model_pool)
        
        result = agent.analyze_image(
            str(fake_image),
            question="¿Qué hay en esta imagen?"
        )
        
        assert "text" in result
        assert len(result["text"]) > 0
        assert "metadata" in result
        assert result["metadata"]["model"] == "qwen3_vl_4b"
        
        # Verificar que se cargó el modelo
        mock_model_pool.get.assert_called_once_with("qwen3_vl_4b")
    
    def test_analyze_image_from_bytes(self, mock_model_pool):
        """Test V2: Analizar imagen desde bytes"""
        agent = VisionAgent(mock_model_pool)
        
        fake_bytes = b"fake_image_bytes"
        
        result = agent.analyze_image(
            fake_bytes,
            question="Describe esta imagen"
        )
        
        assert "text" in result
        assert result["metadata"]["question"] == "Describe esta imagen"
    
    def test_describe_diagram_helper(self, mock_model_pool, tmp_path):
        """Test V3: Helper describe_diagram()"""
        fake_diagram = tmp_path / "diagram.png"
        fake_diagram.write_bytes(b"fake_diagram")
        
        agent = VisionAgent(mock_model_pool)
        
        description = agent.describe_diagram(str(fake_diagram))
        
        assert isinstance(description, str)
        assert len(description) > 0
    
    def test_extract_text_ocr_helper(self, mock_model_pool, tmp_path):
        """Test V4: Helper extract_text_ocr()"""
        fake_text_image = tmp_path / "text.png"
        fake_text_image.write_bytes(b"fake_text_image")
        
        agent = VisionAgent(mock_model_pool)
        
        # Mock respuesta OCR
        mock_model_pool.get.return_value.create_completion = lambda p, **kw: {
            "choices": [{"text": "Texto extraído de la imagen"}]
        }
        
        text = agent.extract_text_ocr(str(fake_text_image))
        
        assert isinstance(text, str)
        assert len(text) > 0


class TestMemoryManagement:
    """Tests de gestión de memoria automática"""
    
    @pytest.fixture
    def mock_model_pool(self):
        pool = MagicMock()
        vision_model = MagicMock()
        vision_model.create_completion = lambda p, **kw: {
            "choices": [{"text": "Respuesta"}]
        }
        pool.get.return_value = vision_model
        return pool
    
    @patch('psutil.virtual_memory')
    def test_auto_release_on_low_ram(self, mock_psutil, mock_model_pool, tmp_path):
        """Test V5: Auto-liberación cuando RAM < 4GB"""
        # Simular RAM baja (3GB disponible)
        mock_memory = MagicMock()
        mock_memory.available = 3 * (1024**3)  # 3 GB
        mock_psutil.return_value = mock_memory
        
        agent = VisionAgent(mock_model_pool)
        
        fake_image = tmp_path / "test.png"
        fake_image.write_bytes(b"fake")
        
        agent.analyze_image(str(fake_image))
        
        # Debe liberar modelo automáticamente
        mock_model_pool.release.assert_called_once_with("qwen3_vl_4b")
    
    @patch('psutil.virtual_memory')
    def test_no_release_on_sufficient_ram(self, mock_psutil, mock_model_pool, tmp_path):
        """Test V6: NO liberar si RAM > 4GB"""
        # Simular RAM suficiente (8GB disponible)
        mock_memory = MagicMock()
        mock_memory.available = 8 * (1024**3)  # 8 GB
        mock_psutil.return_value = mock_memory
        
        agent = VisionAgent(mock_model_pool)
        
        fake_image = tmp_path / "test.png"
        fake_image.write_bytes(b"fake")
        
        agent.analyze_image(str(fake_image))
        
        # NO debe liberar modelo
        mock_model_pool.release.assert_not_called()


class TestErrorHandling:
    """Tests de manejo de errores"""
    
    @pytest.fixture
    def mock_model_pool(self):
        pool = MagicMock()
        vision_model = MagicMock()
        vision_model.create_completion = lambda p, **kw: {
            "choices": [{"text": "OK"}]
        }
        pool.get.return_value = vision_model
        return pool
    
    def test_image_not_found(self, mock_model_pool):
        """Test V7: Error si imagen no existe"""
        agent = VisionAgent(mock_model_pool)
        
        with pytest.raises(FileNotFoundError):
            agent.analyze_image("/path/to/nonexistent/image.png")
    
    def test_model_error_releases_memory(self, mock_model_pool, tmp_path):
        """Test V8: Libera memoria si modelo falla"""
        # Simular error en modelo
        mock_model_pool.get.return_value.create_completion = MagicMock(
            side_effect=Exception("Model error")
        )
        
        agent = VisionAgent(mock_model_pool)
        
        fake_image = tmp_path / "test.png"
        fake_image.write_bytes(b"fake")
        
        with pytest.raises(RuntimeError, match="Error en análisis de imagen"):
            agent.analyze_image(str(fake_image))
        
        # Debe liberar modelo incluso con error
        mock_model_pool.release.assert_called_once_with("qwen3_vl_4b")
    
    def test_video_analysis_not_implemented(self, mock_model_pool):
        """Test V9: Video analysis placeholder"""
        agent = VisionAgent(mock_model_pool)
        
        with pytest.raises(NotImplementedError, match="opencv-python"):
            agent.analyze_video("video.mp4")


class TestFactory:
    """Tests de factory function"""
    
    def test_create_vision_agent(self):
        """Test V10: Factory crea agente correctamente"""
        mock_pool = MagicMock()
        
        agent = create_vision_agent(mock_pool)
        
        assert isinstance(agent, VisionAgent)
        assert agent.model_pool is mock_pool
        assert agent.model_name == "qwen3_vl_4b"


@pytest.mark.integration
@pytest.mark.slow
class TestIntegrationWithModelPool:
    """Tests de integración con ModelPool real"""
    
    def test_load_qwen3_vl_with_real_pool(self):
        """
        Test V11: Integración end-to-end con ModelPool real
        
        Requiere:
        - config/sarai.yaml con qwen3_vl_4b configurado
        - Modelo GGUF descargado (3.3 GB)
        - Al menos 8GB RAM disponible
        
        Ejecutar con: pytest -v -m integration
        """
        import os
        from core.model_pool import ModelPool
        
        config_path = "config/sarai.yaml"
        if not os.path.exists(config_path):
            pytest.skip("Config no encontrado")
        
        if os.getenv("CI") == "true":
            pytest.skip("CI - requiere modelo GGUF grande")
        
        try:
            # Cargar ModelPool real
            pool = ModelPool(config_path)
            
            # Crear agente
            agent = create_vision_agent(pool)
            
            # Verificar que puede cargar modelo (sin ejecutar, solo carga)
            model = pool.get("qwen3_vl_4b")
            
            assert model is not None
            
            # Verificar stats
            stats = pool.get_stats()
            assert "qwen3_vl_4b" in stats["models_loaded"] or \
                   len(stats["models_loaded"]) > 0
            
            # Cleanup
            pool.release("qwen3_vl_4b")
            
            print("✅ Test de integración Qwen3-VL exitoso")
        
        except Exception as e:
            pytest.fail(f"Integración falló: {e}")


# ============================================================================
# Test Suite Summary
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
