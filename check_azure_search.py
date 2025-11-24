"""Script para verificar contenido de Azure Search"""
from api.infrastructure.adapters.output.azure_search_adapter import AzureSearchAdapter

def main():
    adapter = AzureSearchAdapter()
    
    # Buscar todos los documentos con nombre_completo
    results = adapter.search_client.search(
        search_text="*",
        select=["nombre_completo", "filename"],
        top=10000
    )
    
    personas = {}
    for result in results:
        nombre = result.get("nombre_completo", "Desconocido")
        filename = result.get("filename", "")
        if nombre not in personas:
            personas[nombre] = []
        if filename and filename not in personas[nombre]:
            personas[nombre].append(filename)
    
    print('=' * 80)
    print('PERSONAS EN AZURE SEARCH')
    print('=' * 80)
    print(f'Total personas: {len(personas)}')
    print()
    
    # Buscar específicamente Jose o Sanchez
    jose_found = False
    for persona in sorted(personas.keys()):
        if 'jose' in persona.lower() or 'sanchez' in persona.lower():
            jose_found = True
            print(f'>>> {persona}')
            for file in personas[persona]:
                print(f'    - {file}')
    
    if not jose_found:
        print('NO SE ENCONTRO "Jose Sanchez" ni variantes')
        print()
        print('Nombres que contienen "Jose":')
        for persona in sorted(personas.keys()):
            if 'jose' in persona.lower():
                print(f'  - {persona}')
        print()
        print('Nombres que contienen "Sanchez":')
        for persona in sorted(personas.keys()):
            if 'sanchez' in persona.lower():
                print(f'  - {persona}')
    
    print()
    print('TODOS LOS NOMBRES (primeros 20):')
    print('-' * 80)
    for i, persona in enumerate(sorted(personas.keys())[:20], 1):
        files_count = len(personas[persona])
        print(f'{i:2d}. {persona} ({files_count} archivo{"s" if files_count > 1 else ""})')
    
    if len(personas) > 20:
        print(f'... y {len(personas) - 20} personas mas')
    
    print('=' * 80)

if __name__ == "__main__":
    main()

