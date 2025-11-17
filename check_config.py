"""
Script para verificar la configuración del proyecto.
Verifica que todas las variables de entorno estén configuradas correctamente.
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))


def check_environment():
    """Verifica la configuración del entorno."""
    print("🔍 Verificando configuración...\n")
    
    errors = []
    warnings = []
    
    # 1. Verificar archivo .env
    env_file = Path(".env")
    if not env_file.exists():
        errors.append("❌ Archivo .env no encontrado")
        print("❌ Archivo .env no encontrado")
        print("   Ejecuta: cp .env.example .env")
        return False
    else:
        print("✅ Archivo .env encontrado")
    
    # 2. Verificar configuración
    try:
        from api.utils.config import settings
        print("✅ Configuración cargada")
    except Exception as e:
        errors.append(f"❌ Error cargando configuración: {str(e)}")
        print(f"❌ Error cargando configuración: {str(e)}")
        return False
    
    # 3. Verificar Azure OpenAI
    print("\n📋 Azure OpenAI:")
    if settings.AZURE_OPENAI_ENDPOINT:
        print(f"   ✅ Endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
    else:
        errors.append("❌ AZURE_OPENAI_ENDPOINT no configurado")
        print("   ❌ Endpoint no configurado")
    
    if settings.AZURE_OPENAI_API_KEY and len(str(settings.AZURE_OPENAI_API_KEY)) > 10:
        api_key_str = str(settings.AZURE_OPENAI_API_KEY)
        print(f"   ✅ API Key: {'*' * 10}...{api_key_str[-4:]}")
    else:
        errors.append("❌ AZURE_OPENAI_API_KEY no configurado")
        print("   ❌ API Key no configurado")
    
    print(f"   ✅ Deployment: {settings.AZURE_OPENAI_DEPLOYMENT_NAME}")
    print(f"   ✅ API Version: {settings.AZURE_OPENAI_API_VERSION}")
    
    # 4. Verificar Azure AI Search
    print("\n📋 Azure AI Search:")
    if settings.AZURE_SEARCH_ENDPOINT:
        print(f"   ✅ Endpoint: {settings.AZURE_SEARCH_ENDPOINT}")
    else:
        warnings.append("⚠️ AZURE_SEARCH_ENDPOINT no configurado")
        print("   ⚠️ Endpoint no configurado (necesario para usar el agente)")
        print("   Ver: AZURE_SETUP.md")
    if settings.AZURE_SEARCH_API_KEY and len(str(settings.AZURE_SEARCH_API_KEY)) > 10:
        search_key_str = str(settings.AZURE_SEARCH_API_KEY)
        print(f"   ✅ API Key: {'*' * 10}...{search_key_str[-4:]}")
    else:
        warnings.append("⚠️ AZURE_SEARCH_API_KEY no configurado")
        print("   ⚠️ API Key no configurado (necesario para usar el agente)")
    
    print(f"   ✅ Index Name: {settings.AZURE_SEARCH_INDEX_NAME}")
    
    # 5. Verificar configuración RAG
    print("\n📋 Configuración RAG:")
    print(f"   ✅ Chunk Size: {settings.CHUNK_SIZE}")
    print(f"   ✅ Chunk Overlap: {settings.CHUNK_OVERLAP}")
    print(f"   ✅ Top K Results: {settings.TOP_K_RESULTS}")
    
    # 6. Verificar directorio de logs
    print("\n📋 Sistema:")
    log_dir = Path("logs")
    if log_dir.exists():
        print("   ✅ Directorio logs/ existe")
    else:
        log_dir.mkdir(exist_ok=True)
        print("   ✅ Directorio logs/ creado")
    
    # 7. Test de importaciones
    print("\n📦 Verificando dependencias:")
    required_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("langchain", "LangChain"),
        ("langchain_openai", "LangChain OpenAI"),
        ("azure.search.documents", "Azure Search"),
        ("pydantic", "Pydantic"),
        ("loguru", "Loguru"),
    ]
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            errors.append(f"❌ {name} no instalado")
            print(f"   ❌ {name} no instalado")
    
    # Resumen
    print("\n" + "="*60)
    if errors:
        print("❌ ERRORES ENCONTRADOS:")
        for error in errors:
            print(f"   {error}")
        print("\nNo puedes ejecutar la aplicación hasta resolver estos errores.")
        return False
    elif warnings:
        print("⚠️ ADVERTENCIAS:")
        for warning in warnings:
            print(f"   {warning}")
        print("\nLa aplicación puede no funcionar completamente.")
        print("Ver AZURE_SETUP.md para configurar Azure AI Search.")
        return True
    else:
        print("✅ CONFIGURACIÓN CORRECTA")
        print("\nPróximos pasos:")
        print("1. python init_index.py    # Inicializar índice")
        print("2. python app.py           # Ejecutar aplicación")
        print("3. http://localhost:8000/docs  # Ver documentación API")
        return True


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
