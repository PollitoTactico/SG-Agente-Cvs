# 🤖 SG Agente CVs - Sistema RAG para Análisis de CVs

Sistema de Retrieval-Augmented Generation (RAG) para análisis inteligente de CVs usando Azure OpenAI, Azure AI Search y Azure Blob Storage.

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                         │
│              SG-Employe-Analisis-Front                      │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTP/REST
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 Backend FastAPI                             │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   RAG       │  │  Document    │  │   Storage    │     │
│  │   Service   │  │   Manager    │  │   Stats      │     │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                  │              │
│         ▼                 ▼                  ▼              │
│  ┌──────────────────────────────────────────────────┐     │
│  │          Azure OpenAI GPT-4o mini               │     │
│  │     (Embeddings + Chat Completions)             │     │
│  └──────────────────────────────────────────────────┘     │
│         │                 │                  │              │
│         ▼                 ▼                  ▼              │
│  ┌─────────────┐   ┌──────────────┐  ┌──────────────┐    │
│  │  Azure AI   │   │ Azure Blob   │  │  Azure Blob  │    │
│  │   Search    │   │   Storage    │  │   Storage    │    │
│  │  (Vectors)  │   │    (PDFs)    │  │ (Embeddings) │    │
│  └─────────────┘   └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Características Principales

- 🔍 **Búsqueda Híbrida**: Combina búsqueda vectorial + keywords para máxima precisión
- 🧠 **Azure OpenAI GPT-4o mini**: Comprensión avanzada de lenguaje natural
- 📊 **Azure AI Search**: Indexación vectorial de alta performance
- 💾 **Azure Blob Storage**: Almacenamiento persistente de PDFs y embeddings
- 🎯 **Detección inteligente**: Distingue entre búsquedas generales vs consultas específicas
- 📈 **Múltiples perfiles**: Retorna información de varios candidatos en búsquedas generales
- 🔐 **CORS configurado**: Listo para frontend React

## 🚀 Instalación

### Requisitos previos

- Python 3.9+
- Cuenta Azure con:
  - Azure OpenAI Service
  - Azure AI Search
  - Azure Blob Storage

### Setup

1. **Clonar el repositorio**
```powershell
git clone <repo-url>
cd SG-Agente-Cvs
```

2. **Crear entorno virtual**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. **Instalar dependencias**
```powershell
pip install -r requirements.txt
```

4. **Configurar variables de entorno**

Editar `.env`:
```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://tu-endpoint.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=tu-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://tu-search.search.windows.net
AZURE_SEARCH_API_KEY=tu-search-key
AZURE_SEARCH_INDEX_NAME=cvs-knowledge-base

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_STORAGE_CONTAINER_PDFS=pdfs
AZURE_STORAGE_CONTAINER_EMBEDDINGS=embeddings
```

5. **Inicializar el índice de Azure Search**
```powershell
python init_index.py
```

6. **Ejecutar el servidor**
```powershell
python app.py
```

El servidor estará disponible en `http://localhost:8000`

## 📚 API Endpoints

### 🔍 RAG Agent

**POST** `/api/v1/query`
```json
{
  "query": "perfiles que sepan C#",
  "session_id": "optional-uuid",
  "filters": {}
}
```

Respuesta:
```json
{
  "answer": "Encontré los siguientes perfiles con conocimientos en C#...",
  "sources": [...],
  "session_id": "uuid",
  "metadata": {
    "documents_found": 15,
    "nombre_buscado": ""
  }
}
```

### 📄 Documents

**POST** `/api/v1/documents/upload`
- Sube un PDF, lo indexa y guarda embeddings

**GET** `/api/v1/documents`
- Lista todos los documentos indexados

**DELETE** `/api/v1/documents/{document_id}`
- Elimina un documento del índice

### 📊 Storage Stats

**GET** `/api/v1/storage/stats`
```json
{
  "azure_search": {
    "total_chunks": 1234,
    "unique_documents": 45,
    "unique_personas": 42
  },
  "azure_blob_storage": {
    "pdfs_count": 45,
    "embeddings_count": 45
  }
}
```

### 🏥 Health Check

**GET** `/health`
- Verifica el estado del sistema

## 🎯 Ejemplos de Uso

### Búsqueda General (Múltiples CVs)
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "perfiles que sepan Python y React"}'
```

### Consulta Específica (Una persona)
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "certificaciones de Juan Pérez"}'
```

### Ver Estadísticas
```bash
curl http://localhost:8000/api/v1/storage/stats
```

## 🧪 Testing

```powershell
# Ejecutar tests
pytest

# Con coverage
pytest --cov=api tests/
```

## 📦 Estructura del Proyecto

```
SG-Agente-Cvs/
├── api/
│   ├── application/           # Lógica de negocio
│   │   ├── input/port/       # Puertos de entrada
│   │   ├── output/port/      # Puertos de salida
│   │   └── service/          # Servicios (RAG, DocumentManager)
│   ├── infrastructure/        # Adaptadores
│   │   └── adapters/
│   │       ├── input/        # FastAPI adapter
│   │       └── output/       # Azure adapters
│   └── utils/                # Config, logger
├── tests/                    # Tests unitarios
├── app.py                    # Punto de entrada
├── requirements.txt
└── .env                      # Variables de entorno
```

## 🔧 Configuración Avanzada

### RAG Configuration

Editar `.env`:
```env
CHUNK_SIZE=1500              # Tamaño de chunks
CHUNK_OVERLAP=300            # Overlap entre chunks
TOP_K_RESULTS=200            # Documentos a recuperar
```

### Prompt System

Personalizar en `api/infrastructure/adapters/output/azure_openai_adapter.py`:
```python
def _get_system_prompt(self) -> str:
    return """Tu prompt personalizado aquí..."""
```

## 🚨 Troubleshooting

### Error: "Azure Search no configurado"
Verificar que `.env` tenga:
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`

### Error: "Modelo no encontrado"
Verificar que el deployment `gpt-4o-mini` exista en Azure OpenAI

### Solo retorna 1 CV cuando debería retornar varios
- Verificar logs: Debe decir "🔍 Búsqueda general detectada"
- Aumentar `TOP_K_RESULTS` en `.env`

## 📈 Monitoreo

Ver logs en tiempo real:
```powershell
tail -f logs/app.log
```

## 🔒 Seguridad

- ✅ API Keys en `.env` (no committear)
- ✅ CORS configurado
- ✅ Validación de tipos con Pydantic
- ✅ Sanitización de inputs

## 📝 Licencia

Proprietary - SG Consulting

## 👥 Equipo

Desarrollado por el equipo de SG Consulting

---

**Swagger UI**: http://localhost:8000/docs  
**ReDoc**: http://localhost:8000/redoc
