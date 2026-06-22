from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import wave

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
RESULTADOS_DIR = ROOT_DIR / "Implementacion_last" / "resultados_punto3_validacion"
REPRESENTACIONES_DIR = RESULTADOS_DIR / "representaciones"
PREDICCIONES_DIR = (
    RESULTADOS_DIR
    / "clasificadores"
    / "UjaNet"
    / "multiclase"
)
METADATOS_PATH = REPRESENTACIONES_DIR / "STFT" / "metadatos.csv"
SEMILLA_ANONIMA = 42
REPRESENTACIONES = (
    "STFT",
    "MFCC",
    "MelSpectrogram",
    "LogMelSpectrogram",
    "DeepONMF_W",
    "DeepONMF_H3",
)
NOMBRES_REPRESENTACIONES = {
    "STFT": "STFT",
    "MFCC": "MFCC",
    "MelSpectrogram": "Mel-Spectrogram",
    "LogMelSpectrogram": "Log-Mel Spectrogram",
    "DeepONMF_W": "Deep-ONMF W (matriz de bases)",
    "DeepONMF_H3": "Deep-ONMF H3 (matriz de activaciones)",
}
NOMBRES_CLASES = {
    "N": "Normal",
    "AS": "Aortic Stenosis",
    "MR": "Mitral Regurgitation",
    "MS": "Mitral Stenosis",
    "MVP": "Mitral Valve Prolapse",
}


@dataclass(frozen=True)
class AudioAnonimo:
    id_publico: int
    etiqueta: str
    indice_interno: int
    duracion_s: float


@dataclass(frozen=True)
class ResultadoPrueba:
    representacion: str
    representacion_titulo: str
    prediccion_multiclase: str
    prediccion_multiclase_nombre: str
    prediccion_binaria: str
    fold: int
    clase_real: str

    @property
    def acierto(self) -> bool:
        return self.prediccion_multiclase == self.clase_real


@dataclass(frozen=True)
class Diagnostico:
    etiqueta_audio: str
    resultados: tuple[ResultadoPrueba, ...]
    clase_real: str
    clase_real_nombre: str
    etiqueta_real_binaria: str
    archivo_original: str
    ruta_original: str

    @property
    def aciertos(self) -> int:
        return sum(1 for resultado in self.resultados if resultado.acierto)


def comprobar_estructura() -> None:
    requeridos = [
        RESULTADOS_DIR,
        REPRESENTACIONES_DIR,
        PREDICCIONES_DIR,
        METADATOS_PATH,
    ]
    for ruta in requeridos:
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontro el recurso necesario: {ruta}")
    for representacion in REPRESENTACIONES:
        ruta = REPRESENTACIONES_DIR / representacion / f"{representacion}.npz"
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontro la representacion: {ruta}")


@lru_cache(maxsize=1)
def metadatos() -> pd.DataFrame:
    comprobar_estructura()
    df = pd.read_csv(METADATOS_PATH)
    if "indice_interno" not in df.columns:
        raise ValueError("metadatos.csv no contiene la columna indice_interno")
    return df.sort_values("indice_interno").reset_index(drop=True)


@lru_cache(maxsize=1)
def mapa_anonimo() -> tuple[int, ...]:
    total = len(metadatos())
    rng = np.random.default_rng(SEMILLA_ANONIMA)
    return tuple(int(i) for i in rng.permutation(total))


def listar_audios() -> list[AudioAnonimo]:
    df = metadatos()
    audios: list[AudioAnonimo] = []
    for posicion, indice in enumerate(mapa_anonimo(), start=1):
        fila = df.iloc[indice]
        audios.append(
            AudioAnonimo(
                id_publico=posicion,
                etiqueta=f"audio{posicion}",
                indice_interno=int(fila["indice_interno"]),
                duracion_s=float(fila["duracion_s"]),
            )
        )
    return audios


def resolver_indice(id_publico: int) -> int:
    mapping = mapa_anonimo()
    if id_publico < 1 or id_publico > len(mapping):
        raise ValueError(f"Audio fuera de rango: audio{id_publico}")
    return int(mapping[id_publico - 1])


def etiqueta_audio(id_publico: int) -> str:
    resolver_indice(id_publico)
    return f"audio{id_publico}"


def fila_audio(id_publico: int) -> pd.Series:
    return metadatos().iloc[resolver_indice(id_publico)]


@lru_cache(maxsize=len(REPRESENTACIONES))
def matriz_representacion(nombre: str) -> np.ndarray:
    if nombre not in REPRESENTACIONES:
        raise ValueError(f"Representacion no soportada: {nombre}")
    ruta = REPRESENTACIONES_DIR / nombre / f"{nombre}.npz"
    with np.load(ruta) as datos:
        return datos["x"].astype(np.float32, copy=False)


def representaciones_audio(id_publico: int) -> dict[str, np.ndarray]:
    indice = resolver_indice(id_publico)
    return {nombre: matriz_representacion(nombre)[indice] for nombre in REPRESENTACIONES}


def leer_wav_normalizado(id_publico: int) -> tuple[np.ndarray, int]:
    ruta = Path(str(fila_audio(id_publico)["ruta"]))
    with wave.open(str(ruta), "rb") as wav:
        canales = wav.getnchannels()
        frecuencia = wav.getframerate()
        bytes_muestra = wav.getsampwidth()
        bruto = wav.readframes(wav.getnframes())

    if bytes_muestra == 1:
        datos = np.frombuffer(bruto, dtype=np.uint8).astype(np.float64)
        datos = (datos - 128.0) / 128.0
    elif bytes_muestra == 2:
        datos = np.frombuffer(bruto, dtype=np.int16).astype(np.float64) / 32768.0
    elif bytes_muestra == 4:
        datos = np.frombuffer(bruto, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Formato WAV no soportado: {bytes_muestra} bytes por muestra")

    if canales > 1:
        datos = datos.reshape(-1, canales).mean(axis=1)
    return datos.astype(np.float64, copy=False), int(frecuencia)


@lru_cache(maxsize=len(REPRESENTACIONES))
def predicciones_ujanet(representacion: str) -> pd.DataFrame:
    if representacion not in REPRESENTACIONES:
        raise ValueError(f"Representacion no soportada para UjaNet: {representacion}")
    carpeta = PREDICCIONES_DIR / representacion
    frames = []
    for ruta in sorted(carpeta.glob("fold_*_predicciones.csv")):
        frames.append(pd.read_csv(ruta))
    if not frames:
        raise FileNotFoundError(f"No se encontraron predicciones en {carpeta}")
    df = pd.concat(frames, ignore_index=True)
    df["_ruta_norm"] = df["ruta"].astype(str).str.lower()
    return df


def _resultado_representacion(
    representacion: str,
    ruta_norm: str,
    archivo: str,
    clase_real: str,
) -> ResultadoPrueba:
    predicciones = predicciones_ujanet(representacion)
    coincidencias = predicciones[predicciones["_ruta_norm"] == ruta_norm]
    if coincidencias.empty:
        coincidencias = predicciones[predicciones["archivo"].astype(str) == archivo]
    if coincidencias.empty:
        raise LookupError(f"No hay prediccion UjaNet para {archivo} con {representacion}")

    pred = coincidencias.iloc[0]
    pred_multi = str(pred["pred_multiclase"])
    return ResultadoPrueba(
        representacion=representacion,
        representacion_titulo=NOMBRES_REPRESENTACIONES[representacion],
        prediccion_multiclase=pred_multi,
        prediccion_multiclase_nombre=NOMBRES_CLASES.get(pred_multi, pred_multi),
        prediccion_binaria=str(pred["pred_binaria"]),
        fold=int(pred["fold"]),
        clase_real=clase_real,
    )


def diagnostico_audio(id_publico: int) -> Diagnostico:
    fila = fila_audio(id_publico)
    ruta_norm = str(fila["ruta"]).lower()
    archivo = str(fila["archivo"])
    clase_real = str(fila["clase"])
    resultados = tuple(
        _resultado_representacion(representacion, ruta_norm, archivo, clase_real)
        for representacion in REPRESENTACIONES
    )
    return Diagnostico(
        etiqueta_audio=etiqueta_audio(id_publico),
        resultados=resultados,
        clase_real=clase_real,
        clase_real_nombre=NOMBRES_CLASES.get(clase_real, clase_real),
        etiqueta_real_binaria=str(fila["etiqueta_binaria"]),
        archivo_original=archivo,
        ruta_original=str(fila["ruta"]),
    )


def ruta_audio(id_publico: int) -> Path:
    return Path(str(fila_audio(id_publico)["ruta"]))
