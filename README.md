# Agente RAG con Arquitectura Hexagonal

Este proyecto implementa un **Agente RAG (Retrieval-Augmented Generation)** usando **arquitectura hexagonal** con **LangChain**, **Azure OpenAI** y **Azure AI Search**.

## ✨ Características

- 🏗️ **Arquitectura Hexagonal**: Separación clara entre dominio e infraestructura
- 🤖 **Azure OpenAI**: GPT-5 para generación de respuestas
- 🔍 **Azure AI Search**: Base de datos vectorial para PDFs
- 📄 **Procesamiento de PDFs**: Indexación automática de documentos
- 💬 **Historial de Conversación**: Contexto de sesión para consultas
- 🚀 **FastAPI**: API REST moderna y rápida
- 🧪 **Testing**: Pruebas unitarias e integración
- 📊 **Logging**: Sistema de logs estructurado

## 🏗️ Arquitectura

```
SG-Agente-Cvs/
├── api/
│   ├── application/          # ⚙️ Capa de Aplicación (Dominio)
│   │   ├── input/port/      # 📥 Interfaces de entrada
│   │   │   ├── rag_agent_port.py
│   │   │   └── document_manager_port.py
│   │   ├── output/port/     # 📤 Interfaces de salida
│   │   │   ├── llm_port.py
│   │   │   └── vector_store_port.py
│   │   └── service/         # 💼 Lógica de negocio
│   │       ├── rag_agent_service.py
│   │       └── document_manager_service.py
│   ├── infrastructure/       # 🔧 Capa de Infraestructura
│   │   └── adapters/
│   │       ├── input/       # 🌐 Adaptadores de entrada
│   │       │   ├── fastapi_adapter.py
│   │       │   └── models.py
│   │       └── output/      # 🔌 Adaptadores de salida
│   │           ├── azure_openai_adapter.py
│   │           └── azure_search_adapter.py
│   └── utils/               # 🛠️ Utilidades
│       ├── config.py
│       └── logger.py
├── logs/                    # 📝 Logs de la aplicación
├── tests/                   # 🧪 Pruebas
├── app.py                   # 🚀 Punto de entrada
├── init_index.py           # 🔧 Script de inicialización
└── example.py              # 📖 Ejemplo de uso
```

### Flujo de Datos (RAG)

```
Usuario
  ↓
FastAPI (Adaptador Input)
  ↓
RAG Service (Dominio)
  ↓
┌─────────────────┬──────────────────┐
↓                 ↓                  ↓
LLM Port    Vector Store Port   Session History
↓                 ↓
Azure OpenAI   Azure AI Search
↓                 ↓
Embeddings    Similarity Search
  ↓                 ↓
  └─────→ Respuesta + Fuentes ←─────┘
            ↓
       Usuario
```

## 🚀 Inicio Rápido

### Prerequisitos
- Python 3.11+
- Cuenta de Azure activa
- Azure OpenAI deployment (ya configurado)

### Instalación

1. **Clonar y preparar entorno**
   ```powershell
   cd c:\Programacion\SG-Agente-Cvs
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Instalar dependencias**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configurar Azure AI Search**
   
   Ver guía detallada: [AZURE_SETUP.md](AZURE_SETUP.md)
   
   Resumen rápido:
   ```powershell
   az search service create `
     --name sg-agente-search `
     --resource-group [TU_GRUPO] `
     --sku free `
     --location eastus
   ```

4. **Configurar variables de entorno**
   ```powershell
   cp .env.example .env
   # Editar .env con tus credenciales
   ```

5. **Inicializar índice vectorial**
   ```powershell
   python init_index.py
   ```

6. **Ejecutar la aplicación**
   ```powershell
   python app.py
   ```

   La API estará en: http://localhost:8000/docs

## 📚 Documentación

- 📖 [QUICKSTART.md](QUICKSTART.md) - Guía de inicio rápido
- ☁️ [AZURE_SETUP.md](AZURE_SETUP.md) - Configuración de Azure AI Search
- 💻 [SCRIPTS.md](SCRIPTS.md) - Scripts útiles de PowerShell

## 🔑 API Endpoints

### Consultas RAG

**POST** `/api/v1/query`
```json
{
  "query": "¿Qué es la arquitectura hexagonal?",
  "session_id": "user-123",
  "filters": {}
}
```

**DELETE** `/api/v1/sessions/{session_id}` - Limpiar historial

### Gestión de Documentos

**POST** `/api/v1/documents/upload` - Subir PDF (multipart/form-data)

**GET** `/api/v1/documents` - Listar documentos

**DELETE** `/api/v1/documents/{document_id}` - Eliminar documento

### Utilidad

**GET** `/health` - Health check

## 🧪 Testing

```powershell
# Todos los tests
pytest

# Con coverage
pytest --cov=api --cov-report=html

# Solo unitarios
pytest -m unit

# Solo integración
pytest -m integration
```

## 💰 Costos de Azure (Optimizado)

### Configuración Recomendada
- **Azure AI Search (Free)**: $0/mes
- **Azure OpenAI**: ~$0.002 por 1K tokens
- **Estimado mensual**: $5-15 en desarrollo

Ver [AZURE_SETUP.md](AZURE_SETUP.md) para detalles.

## 🐳 Docker (Opcional)

```powershell
# Build y ejecutar
docker-compose up -d

# Ver logs
docker-compose logs -f
```

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles

## 🙏 Agradecimientos

- [LangChain](https://github.com/langchain-ai/langchain)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Azure AI Services](https://azure.microsoft.com/services/cognitive-services/)

---

**Nota**: Este proyecto está configurado para consumir la **menor cantidad de recursos de Azure** posible, utilizando el tier **Free** de Azure AI Search y optimizaciones en el uso de tokens.
