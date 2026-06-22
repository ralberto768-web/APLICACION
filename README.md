# Aplicacion local UjaNet

Aplicacion Flask portable para seleccionar audios anonimos de la base Yaseen, visualizar sus representaciones y mostrar el diagnostico UjaNet validado.

La carpeta incluye los datos necesarios para funcionar tras descargar el repositorio. En el primer arranque, si no existe `datos_app/audios_yaseen.zip`, la app lo reconstruye automaticamente a partir de los archivos `datos_app/audios_yaseen.zip.b64.partXXX`.

## Ejecutar

En Windows, doble clic en:

```text
ejecutar_app.bat
```

Ese archivo crea un entorno local, instala los requisitos y arranca la aplicacion.

Ejecucion manual:

```powershell
cd APLICACION
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Despues abre:

```text
http://127.0.0.1:5000
```

Si el puerto 5000 esta ocupado:

```powershell
$env:PORT='5001'
python app.py
```

La verdad original del audio se mantiene oculta hasta pulsar `Validar` en la pantalla de diagnostico.
