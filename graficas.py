from __future__ import annotations

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from servicios_datos import NOMBRES_REPRESENTACIONES, leer_wav_normalizado, representaciones_audio


def _figura_base64(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _limites_robustos(matriz: np.ndarray) -> tuple[float, float]:
    valores = np.asarray(matriz, dtype=float)
    bajo, alto = np.percentile(valores, [2, 98])
    if not np.isfinite(bajo) or not np.isfinite(alto) or bajo == alto:
        bajo = float(np.nanmin(valores))
        alto = float(np.nanmax(valores))
    if bajo == alto:
        alto = bajo + 1e-9
    return float(bajo), float(alto)


def grafica_tiempo(id_publico: int) -> str:
    senal, frecuencia = leer_wav_normalizado(id_publico)
    t = np.arange(len(senal), dtype=float) / max(frecuencia, 1)
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.plot(t, senal, color="#1f6f8b", linewidth=0.85)
    ax.set_title("Senal en tiempo")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud normalizada")
    ax.grid(True, alpha=0.25)
    return _figura_base64(fig)


def grafica_frecuencia(id_publico: int) -> str:
    senal, frecuencia = leer_wav_normalizado(id_publico)
    ventana = np.hamming(len(senal))
    espectro = np.abs(np.fft.rfft(senal * ventana))
    frecuencias = np.fft.rfftfreq(len(senal), d=1.0 / max(frecuencia, 1))
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.plot(frecuencias, espectro, color="#8a4baf", linewidth=0.9)
    ax.set_title("Representacion en frecuencia")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud")
    ax.set_xlim(0, frecuencia / 2)
    ax.grid(True, alpha=0.25)
    return _figura_base64(fig)


def grafica_matriz(titulo: str, matriz: np.ndarray) -> str:
    vmin, vmax = _limites_robustos(matriz)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    imagen = ax.imshow(matriz, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title(titulo)
    ax.set_xlabel("Tramas temporales")
    ax.set_ylabel("Componentes / frecuencia")
    fig.colorbar(imagen, ax=ax, fraction=0.045, pad=0.035)
    return _figura_base64(fig)


def paquete_graficas(id_publico: int) -> dict[str, object]:
    representaciones = representaciones_audio(id_publico)
    matrices = []
    for nombre, matriz in representaciones.items():
        matrices.append(
            {
                "nombre": nombre,
                "titulo": NOMBRES_REPRESENTACIONES[nombre],
                "forma": " x ".join(str(d) for d in matriz.shape),
                "imagen": grafica_matriz(NOMBRES_REPRESENTACIONES[nombre], matriz),
            }
        )
    return {
        "tiempo": grafica_tiempo(id_publico),
        "frecuencia": grafica_frecuencia(id_publico),
        "matrices": matrices,
    }
