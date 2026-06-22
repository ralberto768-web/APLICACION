# Aplicacion local UjaNet

Aplicacion Flask para seleccionar audios anonimos de la base Yaseen, visualizar sus representaciones y mostrar el diagnostico UjaNet validado.

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
