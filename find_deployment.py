"""Script para encontrar el deployment correcto."""
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_API_KEY')

# Deployments comunes para probar
deployments_to_try = [
    "gpt-35-turbo",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini", 
    "gpt-35-turbo-16k",
    "gpt-4-32k",
    "gpt35turbo",
    "gpt4",
    "chat",
    "gpt-5-chat",  # El que tenías antes
    "chatgpt",
]

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-02-15-preview"
)

print("🔍 Buscando deployment disponible...\n")

for deployment in deployments_to_try:
    try:
        print(f"Probando: {deployment}...", end=" ")
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "OK"}],
            max_tokens=5
        )
        print(f"✅ FUNCIONA!")
        print(f"\n🎉 Deployment encontrado: {deployment}")
        print(f"   Tokens disponibles: SÍ")
        print(f"\n📝 Actualiza tu .env con:")
        print(f"   AZURE_OPENAI_DEPLOYMENT_NAME={deployment}")
        break
    except Exception as e:
        if "DeploymentNotFound" in str(e):
            print("❌ No existe")
        elif "quota" in str(e).lower() or "rate" in str(e).lower():
            print(f"⚠️  Existe pero sin cuota/tokens")
        else:
            print(f"❌ Error: {str(e)[:50]}")
else:
    print("\n❌ Ningún deployment común funciona.")
    print("\n🔧 Ve a Azure Portal y copia el nombre EXACTO del deployment:")
    print("   https://portal.azure.com → Azure OpenAI → Model deployments")
