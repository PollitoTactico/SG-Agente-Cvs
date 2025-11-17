# Google Drive Credentials

Este directorio debe contener el archivo de credenciales de Google Service Account.

## Configuración:

1. Ve a Google Cloud Console: https://console.cloud.google.com/
2. Crea o selecciona un proyecto
3. Habilita Google Drive API
4. Crea una Service Account (APIs y servicios → Credenciales → Crear credenciales → Cuenta de servicio)
5. Descarga el archivo JSON de credenciales
6. Guárdalo aquí como: `google_drive_credentials.json`
7. Copia el email de la service account (del archivo JSON, campo `client_email`)
8. Comparte tu folder de Google Drive con ese email (permisos de lectura)

## Estructura del archivo JSON:

El archivo debe verse así:
```json
{
  "type": "service_account",
  "project_id": "tu-proyecto-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "rag-agent@tu-proyecto.iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  ...
}
```

**IMPORTANTE:** Este archivo contiene información sensible. Está en `.gitignore` para no subirlo al repositorio.
