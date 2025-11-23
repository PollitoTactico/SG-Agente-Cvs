"""
Script para iniciar la aplicación con configuración flexible.
Detecta automáticamente si usar Azure Search o InMemory Vector Store.
"""
import os
from api.utils.logger import setup_logger

logger = setup_logger(__name__)


def check_configuration():
    """Verifica la configuración y muestra el estado."""
    print("\n" + "="*80)
    print("🔍 VERIFICACIÓN DE CONFIGURACIÓN")
    print("="*80 + "\n")
    
    # Azure OpenAI
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    openai_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    
    if openai_endpoint and openai_key:
        print("✅ Azure OpenAI: CONFIGURADO")
        print(f"   Endpoint: {openai_endpoint}")
    else:
        print("❌ Azure OpenAI: NO CONFIGURADO")
        print("   Configura AZURE_OPENAI_ENDPOINT y AZURE_OPENAI_API_KEY en .env")
        return False
    
    # Azure Search (opcional)
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    search_key = os.getenv("AZURE_SEARCH_API_KEY", "")
    
    if search_endpoint and search_key and search_endpoint != "<TU_AZURE_SEARCH_ENDPOINT>":
        print("✅ Azure AI Search: CONFIGURADO")
        print(f"   Endpoint: {search_endpoint}")
        print("   Modo: Vector Store con Azure Search")
    else:
        print("⚠️  Azure AI Search: NO CONFIGURADO")
        print("   Usando: InMemory Vector Store (datos se pierden al reiniciar)")
        print("   Para persistencia, configura Azure Search en .env")
    
    # Azure Blob Storage
    storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    
    if storage_conn:
        print("✅ Azure Blob Storage: CONFIGURADO")
        print("   Los PDFs y embeddings se guardarán en Blob Storage")
    else:
        print("❌ Azure Blob Storage: NO CONFIGURADO")
        print("   No se podrán persistir PDFs ni embeddings")
        return False
    
    print("\n" + "="*80)
    print("✅ CONFIGURACIÓN VÁLIDA - Iniciando aplicación...")
    print("="*80 + "\n")
    
    return True


def main():
    """Inicia la aplicación."""
    # Cargar variables de entorno
    from dotenv import load_dotenv
    load_dotenv()
    
    # Verificar configuración
    if not check_configuration():
        print("\n❌ ERROR: Configuración incompleta. Revisa el archivo .env")
        print("   Puedes usar .env.example como referencia\n")
        return
    
    # Importar y ejecutar la app
    from app import main as run_app
    run_app()


if __name__ == "__main__":
    main()
