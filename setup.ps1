# Script de Setup Automatizado para RAG Agent
# Ejecutar: .\setup.ps1

Write-Host "🚀 Iniciando setup de RAG Agent..." -ForegroundColor Cyan
Write-Host ""

# Verificar Python
Write-Host "Verificando Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Python instalado: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python no encontrado. Por favor instala Python 3.11+" -ForegroundColor Red
    exit 1
}

# Crear entorno virtual
Write-Host ""
Write-Host "Creando entorno virtual..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠️  Entorno virtual ya existe" -ForegroundColor Yellow
    $response = Read-Host "¿Deseas recrearlo? (s/n)"
    if ($response -eq "s") {
        Remove-Item -Recurse -Force venv
        python -m venv venv
        Write-Host "✅ Entorno virtual recreado" -ForegroundColor Green
    }
} else {
    python -m venv venv
    Write-Host "✅ Entorno virtual creado" -ForegroundColor Green
}

# Activar entorno virtual
Write-Host ""
Write-Host "Activando entorno virtual..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Instalar dependencias
Write-Host ""
Write-Host "Instalando dependencias (esto puede tardar 2-3 minutos)..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencias instaladas" -ForegroundColor Green
} else {
    Write-Host "❌ Error instalando dependencias" -ForegroundColor Red
    exit 1
}

# Configurar .env
Write-Host ""
Write-Host "Configurando variables de entorno..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "✅ Archivo .env creado desde .env.example" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  IMPORTANTE: Edita el archivo .env con tus credenciales de Azure" -ForegroundColor Yellow
    Write-Host "   Necesitas configurar:" -ForegroundColor Yellow
    Write-Host "   - AZURE_SEARCH_ENDPOINT" -ForegroundColor Yellow
    Write-Host "   - AZURE_SEARCH_API_KEY" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "¿Deseas abrir .env ahora para editarlo? (s/n)"
    if ($response -eq "s") {
        notepad .env
    }
} else {
    Write-Host "✅ Archivo .env ya existe" -ForegroundColor Green
}

# Crear directorio de logs
Write-Host ""
Write-Host "Creando directorio de logs..." -ForegroundColor Yellow
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs"
    Write-Host "✅ Directorio logs creado" -ForegroundColor Green
} else {
    Write-Host "✅ Directorio logs ya existe" -ForegroundColor Green
}

# Verificar configuración
Write-Host ""
Write-Host "Verificando configuración..." -ForegroundColor Yellow
python check_config.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Green
    Write-Host "✅ SETUP COMPLETADO" -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Green
    Write-Host ""
    Write-Host "Próximos pasos:" -ForegroundColor Cyan
    Write-Host "1. Asegúrate de haber configurado Azure AI Search (ver AZURE_SETUP.md)" -ForegroundColor White
    Write-Host "2. Ejecuta: python init_index.py" -ForegroundColor White
    Write-Host "3. Ejecuta: python app.py" -ForegroundColor White
    Write-Host "4. Abre: http://localhost:8000/docs" -ForegroundColor White
    Write-Host ""
    Write-Host "Para más información, consulta TUTORIAL.md" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "⚠️  Configuración incompleta" -ForegroundColor Yellow
    Write-Host "   Revisa el archivo .env y asegúrate de tener todas las credenciales" -ForegroundColor Yellow
    Write-Host "   Ver AZURE_SETUP.md para instrucciones" -ForegroundColor Yellow
}
