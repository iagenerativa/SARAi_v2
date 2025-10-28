"""
Unit Tests para GPG Signer v2.16
=================================

Tests para core/gpg_signer.py
"""

import pytest
import os
from pathlib import Path
from core.gpg_signer import GPGSigner, get_gpg_signer, SignatureResult

# Skip tests si GPG no está disponible
try:
    import gnupg
    GPG_AVAILABLE = True
except ImportError:
    GPG_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GPG_AVAILABLE,
    reason="python-gnupg not installed"
)


class TestGPGSigner:
    """Unit tests para GPGSigner"""
    
    def test_signer_initialization(self, tmp_path):
        """Test: GPGSigner se inicializa correctamente"""
        signer = GPGSigner(gpg_home=str(tmp_path / "gpg_home"))
        
        assert signer.gpg_home.exists()
        assert signer.gpg is not None
    
    def test_sign_prompt_without_key(self, tmp_path):
        """Test: Firmar sin key configurado retorna error"""
        signer = GPGSigner(key_id=None, gpg_home=str(tmp_path / "gpg_home"))
        
        result = signer.sign_prompt("Test prompt")
        
        assert result.success is False
        assert "No GPG key configured" in result.error_message
    
    @pytest.mark.slow
    def test_generate_key(self, tmp_path):
        """Test: Generación de GPG key"""
        signer = GPGSigner(gpg_home=str(tmp_path / "gpg_home"))
        
        key_id = signer.generate_key(
            name="Test SARAi",
            email="test@localhost"
        )
        
        assert key_id is not None
        assert len(key_id) > 0
    
    @pytest.mark.slow
    def test_sign_and_verify_prompt(self, tmp_path):
        """Test: Firmar y verificar prompt completo"""
        signer = GPGSigner(gpg_home=str(tmp_path / "gpg_home"))
        
        # Generar key
        key_id = signer.generate_key()
        assert key_id is not None
        
        # Actualizar key_id del signer
        signer.key_id = key_id
        
        # Firmar prompt
        prompt = "This is a test reflection prompt"
        result = signer.sign_prompt(prompt)
        
        assert result.success is True
        assert result.signed_content is not None
        assert "BEGIN PGP SIGNED MESSAGE" in result.signed_content
        
        # Verificar firma
        is_valid, verified_key = signer.verify_prompt(result.signed_content)
        
        assert is_valid is True
        assert verified_key is not None
    
    def test_extract_unsigned_content(self, tmp_path):
        """Test: Extraer contenido sin firma"""
        signer = GPGSigner(gpg_home=str(tmp_path / "gpg_home"))
        
        # Mock de signed prompt
        signed_prompt = """-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

This is the original prompt content
-----BEGIN PGP SIGNATURE-----

iQEzBAEBCAAdFiEE... (signature data)
-----END PGP SIGNATURE-----"""
        
        content = signer.extract_unsigned_content(signed_prompt)
        
        assert content == "This is the original prompt content"
    
    def test_singleton_pattern(self):
        """Test: get_gpg_signer retorna singleton"""
        signer1 = get_gpg_signer()
        signer2 = get_gpg_signer()
        
        assert signer1 is signer2


class TestGPGSignerIntegration:
    """Integration tests con OmniLoop"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_signed_reflection_prompt(self, tmp_path):
        """Test: Prompts reflexivos firmados en OmniLoop"""
        signer = GPGSigner(gpg_home=str(tmp_path / "gpg_home"))
        key_id = signer.generate_key()
        signer.key_id = key_id
        
        # Simular prompt reflexivo de OmniLoop
        reflection_template = """
[SYSTEM: Self-Reflection Mode - Iteration 2/3]

Original User Request:
¿Qué es la fotosíntesis?

Your Previous Response (Draft):
La fotosíntesis es un proceso...

INSTRUCTIONS:
1. Analyze your previous response critically
2. Identify factual errors
3. Provide improved version

Improved Response:
"""
        
        # Firmar
        result = signer.sign_prompt(reflection_template)
        
        assert result.success is True
        
        # Verificar
        is_valid, _ = signer.verify_prompt(result.signed_content)
        assert is_valid is True
        
        # Extraer para procesamiento
        unsigned = signer.extract_unsigned_content(result.signed_content)
        assert "Self-Reflection Mode" in unsigned
