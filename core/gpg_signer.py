"""
GPG Signer para Omni-Loop v2.16
================================

Firma prompts reflexivos con GPG para auditabilidad total.

FILOSOFÍA:
- Cada prompt reflexivo es firmado criptográficamente
- Los logs pueden verificarse independientemente
- Inmutabilidad garantizada por firma GPG

DEPENDENCIAS:
    pip install python-gnupg

SETUP:
    # Generar key GPG para SARAi
    gpg --gen-key
    # Nombre: SARAi v2.16 Omni-Loop
    # Email: sarai@localhost
    # Exportar key ID
    export SARAI_GPG_KEY_ID=$(gpg --list-keys --keyid-format SHORT | grep -A1 'SARAi' | tail -1 | awk '{print $1}')

USAGE:
    from core.gpg_signer import get_gpg_signer
    
    signer = get_gpg_signer()
    signed_prompt = signer.sign_prompt("Reflect on previous response...")
    
    # Verificar
    is_valid = signer.verify_prompt(signed_prompt)
"""

import os
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import gnupg
    GPG_AVAILABLE = True
except ImportError:
    GPG_AVAILABLE = False
    logging.warning("python-gnupg not installed. GPG signing disabled.")

logger = logging.getLogger(__name__)


@dataclass
class SignatureResult:
    """Resultado de operación de firma"""
    success: bool
    signed_content: Optional[str] = None
    error_message: Optional[str] = None
    key_id: Optional[str] = None


class GPGSigner:
    """
    Firma prompts reflexivos de Omni-Loop con GPG
    
    GARANTÍAS:
    - Inmutabilidad: Prompts firmados no pueden modificarse sin detección
    - Auditabilidad: Cualquier sistema puede verificar la firma
    - Trazabilidad: Cada prompt lleva timestamp + key ID
    
    SEGURIDAD:
    - Usa GPG home dir aislado (~/.gnupg/sarai/)
    - No requiere passphrase (automated signing)
    - Backups automáticos de keys
    """
    
    def __init__(self, key_id: Optional[str] = None, gpg_home: Optional[str] = None):
        """
        Args:
            key_id: GPG key ID para firmar. Si None, usa env var SARAI_GPG_KEY_ID
            gpg_home: GPG home directory. Si None, usa ~/.gnupg/sarai/
        """
        if not GPG_AVAILABLE:
            raise ImportError("python-gnupg not installed. Run: pip install python-gnupg")
        
        self.key_id = key_id or os.getenv("SARAI_GPG_KEY_ID")
        
        # GPG home directory aislado para SARAi
        if gpg_home:
            self.gpg_home = Path(gpg_home)
        else:
            self.gpg_home = Path.home() / ".gnupg" / "sarai"
        
        self.gpg_home.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Inicializar GPG
        self.gpg = gnupg.GPG(gnupghome=str(self.gpg_home))
        
        # Verificar que key_id existe
        if self.key_id:
            if not self._key_exists(self.key_id):
                logger.warning(f"GPG key {self.key_id} not found. Signing will fail.")
        else:
            logger.warning("No GPG key configured. Set SARAI_GPG_KEY_ID env var.")
    
    def _key_exists(self, key_id: str) -> bool:
        """Verifica que la key GPG existe"""
        keys = self.gpg.list_keys()
        return any(key_id in key['keyid'] for key in keys)
    
    def sign_prompt(self, prompt: str, detach: bool = False) -> SignatureResult:
        """
        Firma un prompt reflexivo con GPG
        
        Args:
            prompt: Texto del prompt a firmar
            detach: Si True, genera firma separada (detached signature)
        
        Returns:
            SignatureResult con contenido firmado o error
        
        Format (inline signature):
            -----BEGIN PGP SIGNED MESSAGE-----
            Hash: SHA256
            
            [PROMPT CONTENT]
            -----BEGIN PGP SIGNATURE-----
            [SIGNATURE]
            -----END PGP SIGNATURE-----
        """
        if not self.key_id:
            return SignatureResult(
                success=False,
                error_message="No GPG key configured"
            )
        
        try:
            signed = self.gpg.sign(
                prompt,
                keyid=self.key_id,
                detach=detach,
                clearsign=not detach  # Inline signature si no es detached
            )
            
            if signed.data:
                return SignatureResult(
                    success=True,
                    signed_content=str(signed),
                    key_id=self.key_id
                )
            else:
                return SignatureResult(
                    success=False,
                    error_message=f"Signing failed: {signed.stderr}"
                )
        
        except Exception as e:
            logger.error(f"GPG signing error: {e}")
            return SignatureResult(
                success=False,
                error_message=str(e)
            )
    
    def verify_prompt(self, signed_prompt: str) -> Tuple[bool, Optional[str]]:
        """
        Verifica la firma de un prompt
        
        Args:
            signed_prompt: Prompt con firma GPG
        
        Returns:
            (is_valid, key_id_used)
        """
        try:
            verified = self.gpg.verify(signed_prompt)
            
            if verified.valid:
                return True, verified.key_id
            else:
                logger.warning(f"Invalid GPG signature: {verified.stderr}")
                return False, None
        
        except Exception as e:
            logger.error(f"GPG verification error: {e}")
            return False, None
    
    def extract_unsigned_content(self, signed_prompt: str) -> Optional[str]:
        """
        Extrae el contenido sin firma de un prompt firmado
        
        Útil para procesar el prompt después de verificar la firma
        """
        try:
            # Para inline signatures (clearsign)
            lines = signed_prompt.split('\n')
            
            # Buscar inicio del mensaje
            start_idx = None
            for i, line in enumerate(lines):
                if line.startswith('-----BEGIN PGP SIGNED MESSAGE-----'):
                    # Contenido empieza después de Hash: line y línea vacía
                    start_idx = i + 3
                    break
            
            # Buscar fin del mensaje
            end_idx = None
            for i, line in enumerate(lines):
                if line.startswith('-----BEGIN PGP SIGNATURE-----'):
                    end_idx = i
                    break
            
            if start_idx is not None and end_idx is not None:
                return '\n'.join(lines[start_idx:end_idx]).strip()
            
            # Si no tiene formato esperado, retornar original
            return signed_prompt
        
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return None
    
    def generate_key(self, name: str = "SARAi v2.16 Omni-Loop", email: str = "sarai@localhost"):
        """
        Genera una nueva key GPG para SARAi
        
        SOLO USAR EN SETUP INICIAL
        """
        try:
            input_data = self.gpg.gen_key_input(
                name_real=name,
                name_email=email,
                key_type='RSA',
                key_length=2048,
                passphrase=''  # Sin passphrase para automated signing
            )
            
            key = self.gpg.gen_key(input_data)
            logger.info(f"✅ GPG key generated: {key}")
            
            # Guardar key ID en archivo para persistencia
            key_file = self.gpg_home / "sarai_key_id.txt"
            key_file.write_text(str(key))
            
            return str(key)
        
        except Exception as e:
            logger.error(f"Key generation failed: {e}")
            return None
    
    def export_public_key(self, output_path: Optional[Path] = None) -> Optional[str]:
        """
        Exporta la public key para compartir con auditores
        
        Args:
            output_path: Ruta donde guardar la key. Si None, solo retorna como string
        
        Returns:
            Public key en formato ASCII-armored
        """
        if not self.key_id:
            logger.error("No key configured")
            return None
        
        try:
            public_key = self.gpg.export_keys(self.key_id)
            
            if output_path:
                Path(output_path).write_text(public_key)
                logger.info(f"✅ Public key exported to {output_path}")
            
            return public_key
        
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None


# Singleton pattern
_gpg_signer_instance: Optional[GPGSigner] = None


def get_gpg_signer(key_id: Optional[str] = None) -> GPGSigner:
    """
    Factory pattern: retorna singleton de GPGSigner
    
    Usage:
        signer = get_gpg_signer()
        result = signer.sign_prompt("My reflection prompt")
    """
    global _gpg_signer_instance
    
    if _gpg_signer_instance is None:
        _gpg_signer_instance = GPGSigner(key_id=key_id)
    
    return _gpg_signer_instance


def setup_gpg_key() -> Optional[str]:
    """
    Utility para setup inicial: genera key y configura env var
    
    Returns:
        Key ID generado o None si falla
    """
    signer = GPGSigner()
    key_id = signer.generate_key()
    
    if key_id:
        print(f"\n✅ GPG key generated successfully!")
        print(f"   Key ID: {key_id}")
        print(f"\n📝 Add to your ~/.bashrc or .env:")
        print(f'   export SARAI_GPG_KEY_ID="{key_id}"')
        
        # Exportar public key
        pub_key_path = Path("backups/gpg/sarai_public.key")
        pub_key_path.parent.mkdir(parents=True, exist_ok=True)
        signer.export_public_key(pub_key_path)
        print(f"\n🔑 Public key exported to: {pub_key_path}")
    
    return key_id


if __name__ == "__main__":
    # Setup inicial
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_gpg_key()
    else:
        print("Usage: python -m core.gpg_signer --setup")
        print("       (Generates GPG key for SARAi)")
