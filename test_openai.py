"""Script para probar Azure OpenAI y verificar tokens disponibles."""
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

def test_connection():
    """Prueba la conexión a Azure OpenAI."""
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    
    print(f"🔍 Probando conexión...")
    print(f"   Endpoint: {endpoint}")
    print(f"   Deployment: {deployment}")
    print()
    
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-02-15-preview"
    )
    
    try:
        print(f"📡 Enviando mensaje de prueba al deployment '{deployment}'...")
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "user", "content": "Responde solo: OK"}
            ],
            max_tokens=10
        )
        
        print(f"✅ ÉXITO! El deployment funciona correctamente")
        print(f"   Respuesta: {response.choices[0].message.content}")
        print(f"   Tokens usados: {response.usage.total_tokens}")
        print(f"   - Prompt: {response.usage.prompt_tokens}")
        print(f"   - Completion: {response.usage.completion_tokens}")
        print()
        print("🎉 Tienes tokens disponibles!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print()
        
        if "DeploymentNotFound" in str(e):
            print("💡 El deployment no existe. Deployments comunes:")
            print("   - gpt-35-turbo")
            print("   - gpt-35-turbo-16k")
            print("   - gpt-4")
            print("   - gpt-4-32k")
            print()
            print("🔧 Verifica el nombre exacto en Azure Portal:")
            print("   Azure OpenAI → Model deployments")
            
        elif "quota" in str(e).lower() or "rate" in str(e).lower():
            print("⚠️  Sin tokens disponibles o límite excedido")
            print("   Verifica la cuota en Azure Portal")
            
        return False

if __name__ == "__main__":
    test_connection()
