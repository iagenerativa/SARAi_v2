#!/usr/bin/env python3
"""
Publica el dashboard de Grafana (grafana_god.json) a Grafana Cloud

Uso en CI/CD:
    GRAFANA_API_KEY=xxx GRAFANA_URL=https://xxx.grafana.net python scripts/publish_grafana.py

Uso local:
    export GRAFANA_API_KEY="glsa_xxx"
    export GRAFANA_URL="https://your-org.grafana.net"
    python scripts/publish_grafana.py
"""
import os
import json
import sys

try:
    import requests
except ImportError:
    print("❌ Error: requests no instalado. Ejecuta: pip install requests")
    sys.exit(1)


def publish_dashboard():
    """Publica grafana_god.json a Grafana Cloud vía API"""
    
    # 1. Validar variables de entorno
    api_key = os.environ.get("GRAFANA_API_KEY")
    base_url = os.environ.get("GRAFANA_URL", "").rstrip('/')
    
    if not api_key:
        print("⚠️  GRAFANA_API_KEY no configurado. Saltando publicación de dashboard.")
        return True  # No es error crítico
    
    if not base_url:
        print("⚠️  GRAFANA_URL no configurado. Saltando publicación de dashboard.")
        return True
    
    # 2. Cargar dashboard JSON
    dashboard_file = "extras/grafana_god.json"
    
    if not os.path.exists(dashboard_file):
        print(f"❌ Error: {dashboard_file} no encontrado")
        return False
    
    print(f"📊 Publicando dashboard desde {dashboard_file}...")
    
    with open(dashboard_file, 'r') as f:
        dashboard_json = json.load(f)
    
    # 3. Preparar payload para API
    payload = {
        "dashboard": dashboard_json,
        "folderId": 0,  # Carpeta "General" (ID 0)
        "overwrite": True,  # Actualiza si ya existe
        "message": f"Published via CI/CD (SARAi v2.6)"
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # 4. POST a Grafana API
    try:
        url = f"{base_url}/api/dashboards/db"
        print(f"🔗 POST {url}")
        
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        
        response.raise_for_status()  # Lanza error si no es 2xx
        
        # 5. Procesar respuesta
        result = response.json()
        dashboard_url = result.get('url', 'N/A')
        dashboard_id = result.get('id', 'N/A')
        
        print(f"✅ Dashboard publicado exitosamente!")
        print(f"   • ID: {dashboard_id}")
        print(f"   • URL: {base_url}{dashboard_url}")
        print(f"   • UID: {result.get('uid', 'N/A')}")
        
        return True
    
    except requests.exceptions.HTTPError as e:
        print(f"❌ Error HTTP al publicar dashboard: {e}")
        print(f"   Respuesta del servidor: {response.text}")
        return False
    
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False


def main():
    """Punto de entrada"""
    print("=" * 60)
    print("SARAi v2.6 - Grafana Dashboard Publisher")
    print("=" * 60)
    
    success = publish_dashboard()
    
    if success:
        print("\n🎉 Publicación completada")
        sys.exit(0)
    else:
        print("\n⚠️  Publicación falló (no crítico en CI/CD)")
        sys.exit(0)  # Exit 0 para no romper el workflow


if __name__ == "__main__":
    main()
