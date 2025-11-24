"""Test completo del RAG con Jose Sanchez"""
import asyncio
from api.application.service.rag_agent_service import RAGAgentService
from api.infrastructure.adapters.output.azure_openai_adapter import AzureOpenAIAdapter
from api.infrastructure.adapters.output.azure_search_adapter import AzureSearchAdapter
from api.application.input.port.rag_agent_port import QueryRequest

async def test_rag():
    print("=" * 80)
    print("PRUEBA COMPLETA RAG: Jose Sanchez")
    print("=" * 80)
    
    # Inicializar servicios
    llm = AzureOpenAIAdapter()
    vector_store = AzureSearchAdapter()
    rag_service = RAGAgentService(llm_port=llm, vector_store_port=vector_store)
    
    # Query
    query = "hablame sobre Jose Sanchez"
    
    print(f"\n1. Query: {query}")
    
    # Crear request
    request = QueryRequest(
        query=query,
        session_id="test-jose-sanchez",
        filters=None
    )
    
    print("\n2. Ejecutando RAG query...")
    response = await rag_service.query(request)
    
    print("\n3. RESPUESTA:")
    print("-" * 80)
    print(response.answer)
    print("-" * 80)
    
    print(f"\n4. METADATA:")
    print(f"   - Documentos encontrados: {response.metadata.get('documents_found', 0)}")
    print(f"   - Documentos iniciales: {response.metadata.get('initial_documents', 0)}")
    print(f"   - Documentos filtrados: {response.metadata.get('filtered_documents', 0)}")
    print(f"   - Nombre buscado: '{response.metadata.get('nombre_buscado', '')}'")
    
    print(f"\n5. SOURCES ({len(response.sources)} fuentes):")
    print("-" * 80)
    
    personas_sources = {}
    for source in response.sources:
        nombre = source.get('nombre_completo', 'Desconocido')
        if nombre not in personas_sources:
            personas_sources[nombre] = []
        personas_sources[nombre].append({
            'seccion': source.get('seccion_cv', 'general'),
            'score': source.get('score', 0),
            'filename': source.get('filename', '')
        })
    
    for persona, sources in personas_sources.items():
        print(f"\n{persona} ({len(sources)} chunks)")
        for s in sources[:3]:
            print(f"  - {s['seccion']} (score: {s['score']:.4f})")
    
    print("\n" + "=" * 80)
    
    # Verificar si Jose Sanchez está en las fuentes
    jose_found = False
    for persona in personas_sources.keys():
        if 'jose' in persona.lower() and 'sanchez' in persona.lower():
            jose_found = True
            print(f"\n✅ JOSE SANCHEZ ENCONTRADO EN SOURCES: {persona}")
            break
    
    if not jose_found:
        print("\n❌ JOSE SANCHEZ NO ESTA EN LAS SOURCES")
        print("Personas que SI están:")
        for p in list(personas_sources.keys())[:5]:
            print(f"  - {p}")

if __name__ == "__main__":
    asyncio.run(test_rag())
