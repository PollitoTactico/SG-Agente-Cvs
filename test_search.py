"""Test de búsqueda para Jose Sanchez"""
import asyncio
from api.infrastructure.adapters.output.azure_openai_adapter import AzureOpenAIAdapter
from api.infrastructure.adapters.output.azure_search_adapter import AzureSearchAdapter

async def test_search():
    llm = AzureOpenAIAdapter()
    search = AzureSearchAdapter()
    
    query = "hablame sobre Jose Sanchez"
    
    print("=" * 80)
    print("PRUEBA DE BUSQUEDA: Jose Sanchez")
    print("=" * 80)
    
    # Generar embedding
    embeddings = await llm.generate_embeddings([query])
    query_embedding = embeddings[0]
    
    print(f"\n1. Query: {query}")
    print(f"2. Embedding generado: {len(query_embedding)} dimensiones")
    
    # Buscar en Azure Search
    print("\n3. Buscando en Azure Search...")
    documents = await search.similarity_search(
        query_embedding=query_embedding,
        top_k=50,
        query_text=query
    )
    
    print(f"\n4. Documentos encontrados: {len(documents)}")
    
    # Mostrar documentos
    personas = {}
    for doc in documents[:20]:  # Primeros 20
        nombre = doc.metadata.get("nombre_completo", "Desconocido")
        if nombre not in personas:
            personas[nombre] = []
        personas[nombre].append({
            "score": doc.score,
            "seccion": doc.metadata.get("seccion_cv", "general"),
            "content_preview": doc.content[:100]
        })
    
    print("\n5. PERSONAS ENCONTRADAS:")
    print("-" * 80)
    for persona, docs in list(personas.items())[:10]:
        print(f"\n{persona} ({len(docs)} chunks)")
        for i, d in enumerate(docs[:2], 1):
            print(f"  {i}. Score: {d['score']:.4f} | Sección: {d['seccion']}")
            print(f"     Preview: {d['content_preview']}...")
    
    # Buscar específicamente Jose o Sanchez
    print("\n\n6. BUSQUEDA ESPECIFICA DE 'JOSE' o 'SANCHEZ':")
    print("-" * 80)
    found_jose = False
    for persona in personas.keys():
        if 'jose' in persona.lower() or 'sanchez' in persona.lower():
            found_jose = True
            print(f">>> ENCONTRADO: {persona}")
            for d in personas[persona][:3]:
                print(f"    Score: {d['score']:.4f} | {d['seccion']}")
    
    if not found_jose:
        print("❌ NO SE ENCONTRO ningún documento de Jose/Sanchez")
        print("\nDocumentos que SI se encontraron:")
        for p in list(personas.keys())[:5]:
            print(f"  - {p}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_search())
