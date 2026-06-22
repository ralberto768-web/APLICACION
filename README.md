# Aplicacion local UjaNet

Aplicacion Flask para seleccionar audios anonimos de la base Yaseen, visualizar sus representaciones y mostrar el diagnostico UjaNet validado.

## Ejecutar

```powershell
cd C:\Users\armga\OneDrive\Escritorio\TFG\APLICACION
& 'C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe' app.py
```

Despues abre:

```text# Aplicacion local UjaNet

Aplicacion Flask portable para seleccionar audios anonimos de la base Yaseen, visualizar sus representaciones y mostrar el diagnostico UjaNet validado.

La carpeta incluye los datos necesarios para funcionar tras descargar el repositorio. En el primer arranque, si no existe `datos_app/audios_yaseen.zip`, la app lo reconstruye automaticamente a partir de los archivos `datos_app/audios_yaseen.zip.b64.partXXX`.

## Ejecutar

```powershell
cd C:\Users\armga\OneDrive\Escritorio\TFG\APLICACION
& 'C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe' app.py
```

Despues abre:

```text
http://127.0.0.1:5000
```

Si el puerto 5000 esta ocupado:

```powershell
$env:PORT='5001'
& 'C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe' app.py
```

La verdad original del audio se mantiene oculta hasta pulsar `Validar` en la pantalla de diagnostico.

http://127.0.0.1:5000
```

Si el puerto 5000 esta ocupado:

```powershell
$env:PORT='5001'
& 'C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe' app.py
```

La verdad original del audio se mantiene oculta hasta pulsar `Validar` en la pantalla de diagnostico.
