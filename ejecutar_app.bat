@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=python"
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON_CMD=py -3"
    ) else (
        echo No se encontro Python en este equipo.
        echo Instala Python 3 y vuelve a ejecutar este archivo.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creando entorno local...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo No se pudo crear el entorno local.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

echo Instalando requisitos...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo No se pudieron instalar los requisitos.
    pause
    exit /b 1
)

echo.
echo Aplicacion lista. Abre http://127.0.0.1:5000 en el navegador.
echo Para cerrar la aplicacion, pulsa Ctrl+C en esta ventana.
echo.
python app.py

endlocal
