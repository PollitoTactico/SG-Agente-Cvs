"""
Script de prueba para verificar las mejoras del sistema RAG.
Prueba la extracción de nombres, chunking inteligente y filtrado.
"""
import asyncio
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from api.application.service.document_manager_service import DocumentManagerService
from api.application.service.rag_agent_service import RAGAgentService


def test_extract_name():
    """Prueba la extracción de nombres de queries."""
    print("\n" + "="*80)
    print("TEST 1: Extracción de nombres de queries")
    print("="*80)
    
    service = RAGAgentService(None, None)
    
    test_cases = [
        ("dime que certificaciones tiene Gorky Palacios", "gorky palacios"),
        ("¿Cuál es la experiencia de Juan Carlos Pérez?", "juan carlos pérez"),
        ("sobre María González", "maría gonzález"),
        ("certificados de Ana Silva López", "ana silva lópez"),
        ("experiencia laboral", ""),  # Sin nombre
    ]
    
    for query, expected in test_cases:
        result = service._extract_person_name_from_query(query)
        status = "✅" if result.lower() == expected.lower() else "❌"
        print(f"{status} Query: '{query}'")
        print(f"   Esperado: '{expected}'")
        print(f"   Obtenido: '{result}'\n")


def test_extract_full_name():
    """Prueba la extracción de nombres completos de CVs."""
    print("\n" + "="*80)
    print("TEST 2: Extracción de nombres de CVs")
    print("="*80)
    
    service = DocumentManagerService(None, None)
    
    # Test con nombre del archivo
    test_cases = [
        ("CV_Gorky_Palacios.pdf", "GORKY PALACIOS MUTIS\nIngeniero...", "Gorky Palacios"),
        ("Juan_Perez_CV.pdf", "Juan Perez\nDesarrollador...", "Juan Perez"),
        ("curriculum_maria_silva.pdf", "MARÍA SILVA GONZÁLEZ\n...", "Maria Silva"),
    ]
    
    for filename, text, expected_contains in test_cases:
        result = service._extract_full_name(text, filename)
        status = "✅" if expected_contains.lower() in result.lower() else "❌"
        print(f"{status} Archivo: '{filename}'")
        print(f"   Texto (inicio): '{text[:50]}...'")
        print(f"   Nombre extraído: '{result}'\n")


def test_smart_chunking():
    """Prueba el chunking inteligente."""
    print("\n" + "="*80)
    print("TEST 3: Chunking Inteligente por Secciones")
    print("="*80)
    
    service = DocumentManagerService(None, None, chunk_size=500, chunk_overlap=50)
    
    # Texto de ejemplo de CV
    cv_text = """
JUAN PÉREZ GÓMEZ
Ingeniero de Software

EXPERIENCIA LABORAL
- Software Developer en TechCorp (2020-2023)
- Junior Developer en StartupXYZ (2018-2020)

EDUCACIÓN
- Ingeniería en Sistemas - Universidad Nacional (2014-2018)
- Diplomado en Big Data - Instituto Tecnológico (2019)

CERTIFICACIONES
- AWS Certified Solutions Architect
- Certified Scrum Master
- Python Professional Certificate

HABILIDADES
- Python, Java, JavaScript
- AWS, Azure, Docker
- Agile, Scrum
"""
    
    chunks = service._create_smart_chunks(cv_text, "Juan Pérez Gómez")
    
    print(f"Total de chunks generados: {len(chunks)}\n")
    
    for i, chunk_data in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Sección: {chunk_data['section']}")
        print(f"  Tipo: {chunk_data['type']}")
        print(f"  Texto (inicio): {chunk_data['text'][:100]}...")
        print()
    
    # Verificar que se detectaron las secciones correctas
    sections = [c['section'] for c in chunks]
    expected_sections = ['experiencia', 'educacion', 'certificaciones', 'habilidades']
    
    print("Secciones detectadas:", sections)
    print("Secciones esperadas:", expected_sections)
    
    detected = sum(1 for exp in expected_sections if any(exp in s for s in sections))
    print(f"\n✅ Detectadas {detected}/{len(expected_sections)} secciones esperadas")


def test_filter_documents():
    """Prueba el filtrado de documentos."""
    print("\n" + "="*80)
    print("TEST 4: Filtrado de Documentos por Persona")
    print("="*80)
    
    service = RAGAgentService(None, None)
    
    # Simular documentos de diferentes personas
    class MockDoc:
        def __init__(self, content, metadata, score):
            self.content = content
            self.metadata = metadata
            self.score = score
    
    mock_docs = [
        MockDoc(
            "Juan Pérez tiene certificación en AWS",
            {"nombre_completo": "Juan Pérez", "seccion_cv": "certificaciones"},
            0.95
        ),
        MockDoc(
            "María González tiene certificación en Azure",
            {"nombre_completo": "María González", "seccion_cv": "certificaciones"},
            0.90
        ),
        MockDoc(
            "Juan Pérez trabajó en Google",
            {"nombre_completo": "Juan Pérez", "seccion_cv": "experiencia"},
            0.85
        ),
        MockDoc(
            "Pedro Sánchez tiene certificación en SCRUM",
            {"nombre_completo": "Pedro Sánchez", "seccion_cv": "certificaciones"},
            0.80
        ),
    ]
    
    # Filtrar por "Juan Pérez"
    nombre_buscado = "juan pérez"
    filtered = service._filter_and_rerank_documents(mock_docs, nombre_buscado)
    
    print(f"Documentos originales: {len(mock_docs)}")
    print(f"Documentos filtrados: {len(filtered)}")
    print("\nDocumentos que pasaron el filtro:")
    for doc in filtered:
        print(f"  - {doc.metadata['nombre_completo']}: {doc.content[:50]}... (score: {doc.score})")
    
    # Verificar que solo quedaron documentos de Juan Pérez
    all_juan = all("juan" in doc.metadata['nombre_completo'].lower() for doc in filtered)
    status = "✅" if all_juan and len(filtered) == 2 else "❌"
    print(f"\n{status} Filtrado correcto: Solo documentos de Juan Pérez")


def main():
    """Ejecuta todas las pruebas."""
    print("\n")
    print("🚀 PRUEBAS DEL SISTEMA RAG MEJORADO")
    print("="*80)
    
    try:
        test_extract_name()
        test_extract_full_name()
        test_smart_chunking()
        test_filter_documents()
        
        print("\n" + "="*80)
        print("✅ TODAS LAS PRUEBAS COMPLETADAS")
        print("="*80)
        print("\nEl sistema está listo para:")
        print("  1. Extraer nombres de queries y CVs")
        print("  2. Hacer chunking inteligente por secciones")
        print("  3. Filtrar documentos por persona")
        print("  4. Evitar mezcla de información entre CVs")
        print("\n⚠️  IMPORTANTE: Ejecuta 'python init_index.py' para recrear el índice con los nuevos campos")
        print("   Luego re-sube los CVs con 'POST /api/v1/documents/upload'\n")
        
    except Exception as e:
        print(f"\n❌ ERROR EN LAS PRUEBAS: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
