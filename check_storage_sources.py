"""
Script para verificar que Azure Search y Blob Storage están correctamente configurados
y contienen los PDFs esperados.
"""
from api.infrastructure.adapters.output.azure_search_adapter import AzureSearchAdapter
from api.infrastructure.adapters.output.azure_blob_adapter import AzureBlobAdapter
from api.utils.config import settings
import json

def check_azure_search():
    """Verifica contenido de Azure Search"""
    print('\n' + '=' * 80)
    print('VERIFICANDO AZURE SEARCH')
    print('=' * 80)
    
    adapter = AzureSearchAdapter()
    
    # Buscar todos los documentos
    results = adapter.search_client.search(
        search_text="*",
        select=["nombre_completo", "filename", "seccion_cv", "document_id"],
        top=10000
    )
    
    personas = {}
    total_chunks = 0
    
    for result in results:
        total_chunks += 1
        nombre = result.get("nombre_completo", "Desconocido")
        filename = result.get("filename", "")
        seccion = result.get("seccion_cv", "")
        doc_id = result.get("document_id", "")
        
        if nombre not in personas:
            personas[nombre] = {
                "archivos": set(),
                "chunks": 0,
                "secciones": set(),
                "doc_ids": set()
            }
        
        personas[nombre]["chunks"] += 1
        if filename:
            personas[nombre]["archivos"].add(filename)
        if seccion:
            personas[nombre]["secciones"].add(seccion)
        if doc_id:
            personas[nombre]["doc_ids"].add(doc_id)
    
    print(f'\n📊 ESTADÍSTICAS:')
    print(f'   Total chunks indexados: {total_chunks}')
    print(f'   Total personas únicas: {len(personas)}')
    print(f'   Promedio chunks/persona: {total_chunks / len(personas) if personas else 0:.1f}')
    
    print(f'\n👥 PRIMERAS 10 PERSONAS:')
    print('-' * 80)
    for i, (persona, data) in enumerate(sorted(personas.items())[:10], 1):
        archivos_str = ', '.join(list(data["archivos"])[:2])
        if len(data["archivos"]) > 2:
            archivos_str += f' (+{len(data["archivos"])-2} más)'
        print(f'{i:2d}. {persona}')
        print(f'    Chunks: {data["chunks"]} | Archivos: {len(data["archivos"])} | Secciones: {len(data["secciones"])}')
        print(f'    Archivos: {archivos_str}')
    
    # Buscar José Sánchez específicamente
    print(f'\n🔍 BÚSQUEDA: "José Sánchez"')
    print('-' * 80)
    jose_found = False
    for persona, data in personas.items():
        if 'jose' in persona.lower() and 'sanchez' in persona.lower():
            jose_found = True
            print(f'✅ ENCONTRADO: {persona}')
            print(f'   Chunks: {data["chunks"]}')
            print(f'   Archivos: {", ".join(data["archivos"])}')
            print(f'   Secciones: {", ".join(data["secciones"])}')
    
    if not jose_found:
        print('❌ NO se encontró "José Sánchez"')
        print('\nBuscando nombres similares:')
        for persona in sorted(personas.keys()):
            if 'jose' in persona.lower() or 'sanchez' in persona.lower():
                print(f'   - {persona}')
    
    return personas

def check_blob_storage():
    """Verifica contenido de Blob Storage"""
    print('\n' + '=' * 80)
    print('VERIFICANDO AZURE BLOB STORAGE')
    print('=' * 80)
    
    adapter = AzureBlobAdapter(
        connection_string=settings.AZURE_STORAGE_CONNECTION_STRING,
        container_pdfs=settings.AZURE_STORAGE_CONTAINER_PDFS,
        container_embeddings=settings.AZURE_STORAGE_CONTAINER_EMBEDDINGS,
        container_cache=settings.AZURE_STORAGE_CONTAINER_CACHE
    )
    
    # Listar PDFs
    pdfs = adapter.list_pdfs()
    print(f'\n📄 CONTENEDOR DE PDFs: {settings.AZURE_STORAGE_CONTAINER_PDFS}')
    print(f'   Total PDFs: {len(pdfs)}')
    
    if pdfs:
        print(f'\n   Primeros 10 PDFs:')
        for i, pdf in enumerate(pdfs[:10], 1):
            size_mb = pdf['size'] / (1024 * 1024)
            print(f'   {i:2d}. {pdf["name"]} ({size_mb:.2f} MB)')
        if len(pdfs) > 10:
            print(f'   ... y {len(pdfs) - 10} PDFs más')
    else:
        print('   ⚠️ No hay PDFs en Blob Storage')
    
    # Verificar embeddings
    doc_ids = adapter.list_all_documents()
    print(f'\n💾 CONTENEDOR DE EMBEDDINGS: {settings.AZURE_STORAGE_CONTAINER_EMBEDDINGS}')
    print(f'   Total documentos con embeddings: {len(doc_ids)}')
    
    if doc_ids:
        print(f'\n   Primeros 10 document IDs:')
        for i, doc_id in enumerate(doc_ids[:10], 1):
            print(f'   {i:2d}. {doc_id}')
        if len(doc_ids) > 10:
            print(f'   ... y {len(doc_ids) - 10} más')
    else:
        print('   ⚠️ No hay embeddings en Blob Storage')
    
    return pdfs, doc_ids

def verify_consistency(personas, pdfs, doc_ids):
    """Verifica consistencia entre Azure Search y Blob Storage"""
    print('\n' + '=' * 80)
    print('VERIFICACIÓN DE CONSISTENCIA')
    print('=' * 80)
    
    print(f'\n📊 RESUMEN:')
    print(f'   Personas en Azure Search: {len(personas)}')
    print(f'   PDFs en Blob Storage: {len(pdfs)}')
    print(f'   Embeddings en Blob Storage: {len(doc_ids)}')
    
    # Verificar si los document_ids de Search coinciden con Blob
    search_doc_ids = set()
    for persona_data in personas.values():
        search_doc_ids.update(persona_data["doc_ids"])
    
    blob_doc_ids = set(doc_ids)
    
    print(f'\n🔗 DOCUMENT IDs:')
    print(f'   En Azure Search: {len(search_doc_ids)}')
    print(f'   En Blob Storage: {len(blob_doc_ids)}')
    
    missing_in_blob = search_doc_ids - blob_doc_ids
    missing_in_search = blob_doc_ids - search_doc_ids
    
    if missing_in_blob:
        print(f'\n   ⚠️ {len(missing_in_blob)} doc_ids en Search pero NO en Blob:')
        for doc_id in list(missing_in_blob)[:5]:
            print(f'      - {doc_id}')
        if len(missing_in_blob) > 5:
            print(f'      ... y {len(missing_in_blob) - 5} más')
    
    if missing_in_search:
        print(f'\n   ⚠️ {len(missing_in_search)} doc_ids en Blob pero NO en Search:')
        for doc_id in list(missing_in_search)[:5]:
            print(f'      - {doc_id}')
        if len(missing_in_search) > 5:
            print(f'      ... y {len(missing_in_search) - 5} más')
    
    if not missing_in_blob and not missing_in_search:
        print('   ✅ Todos los document_ids están sincronizados')
    
    print('\n' + '=' * 80)

def main():
    print('🔍 VERIFICACIÓN DE FUENTES DE DATOS')
    print('Verificando Azure Search y Blob Storage...\n')
    
    try:
        # Verificar Azure Search
        personas = check_azure_search()
        
        # Verificar Blob Storage
        pdfs, doc_ids = check_blob_storage()
        
        # Verificar consistencia
        verify_consistency(personas, pdfs, doc_ids)
        
        print('\n✅ VERIFICACIÓN COMPLETADA')
        
    except Exception as e:
        print(f'\n❌ ERROR: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
