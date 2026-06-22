from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import base64
import wave
import zipfile

import numpy as np
import pandas as pd
from scipy.fftpack import dct


APP_DIR = Path(__file__).resolve().parent
DATOS_DIR = APP_DIR / "datos_app"
AUDIO_ZIP = DATOS_DIR / "audios_yaseen.zip"
AUDIO_PARTES = "audios_yaseen.zip.b64.part*"
METADATOS_PATH = DATOS_DIR / "metadatos.csv"
PREDICCIONES_DIR = DATOS_DIR / "predicciones_ujanet_multiclase"
SEMILLA_ANONIMA = 42
FRECUENCIA_OBJETIVO_HZ = 8000
VENTANA_STFT_MUESTRAS = 150
SALTO_STFT_MUESTRAS = 75
PUNTOS_FFT = 250
BANDAS_MEL = 40
COEFICIENTES_MFCC = 13
RANGOS_DEEP_ONMF = (9, 8, 7)
ITERACIONES_ONMF = 60
PENALIZACION_ORTOGONAL = 0.05
EPS = 1e-12
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


def _reconstruir_zip_si_hace_falta() -> None:
    if AUDIO_ZIP.exists():
        return
    partes = sorted(DATOS_DIR.glob(AUDIO_PARTES))
    if not partes:
        raise FileNotFoundError(
            "No se encontro audios_yaseen.zip ni sus partes audios_yaseen.zip.b64.partXXX."
        )
    datos_b64 = "".join(parte.read_text(encoding="ascii") for parte in partes)
    AUDIO_ZIP.write_bytes(base64.b64decode(datos_b64.encode("ascii")))


def comprobar_estructura() -> None:
    if not DATOS_DIR.exists():
        raise FileNotFoundError(f"No se encontro la carpeta de datos portable: {DATOS_DIR}")
    if not METADATOS_PATH.exists():
        raise FileNotFoundError(f"No se encontro {METADATOS_PATH}")
    if not PREDICCIONES_DIR.exists():
        raise FileNotFoundError(f"No se encontro {PREDICCIONES_DIR}")
    _reconstruir_zip_si_hace_falta()
    for representacion in REPRESENTACIONES:
        carpeta = PREDICCIONES_DIR / representacion
        if not list(carpeta.glob("fold_*_predicciones.csv")):
            raise FileNotFoundError(f"No hay predicciones UjaNet para {representacion}")


@lru_cache(maxsize=1)
def metadatos() -> pd.DataFrame:
    comprobar_estructura()
    df = pd.read_csv(METADATOS_PATH)
    return df.sort_values("indice_interno").reset_index(drop=True)


@lru_cache(maxsize=1)
def mapa_anonimo() -> tuple[int, ...]:
    total = len(metadatos())
    rng = np.random.default_rng(SEMILLA_ANONIMA)
    return tuple(int(i) for i in rng.permutation(total))


def listar_audios() -> list[AudioAnonimo]:
    df = metadatos()
    return [
        AudioAnonimo(
            id_publico=posicion,
            etiqueta=f"audio{posicion}",
            indice_interno=int(df.iloc[indice]["indice_interno"]),
            duracion_s=float(df.iloc[indice]["duracion_s"]),
        )
        for posicion, indice in enumerate(mapa_anonimo(), start=1)
    ]


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


def bytes_audio(id_publico: int) -> bytes:
    fila = fila_audio(id_publico)
    _reconstruir_zip_si_hace_falta()
    with zipfile.ZipFile(AUDIO_ZIP) as zf:
        return zf.read(str(fila["zip_path"]))


def leer_wav_normalizado(id_publico: int) -> tuple[np.ndarray, int]:
    with wave.open(BytesIO(bytes_audio(id_publico)), "rb") as wav:
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


def _dividir_en_tramas(senal: np.ndarray) -> list[np.ndarray]:
    longitud = int(round(2.0 * FRECUENCIA_OBJETIVO_HZ))
    salto = int(round(1.0 * FRECUENCIA_OBJETIVO_HZ))
    if len(senal) < longitud:
        senal = np.pad(senal, (0, longitud - len(senal)), mode="constant")
    tramas = [senal[inicio : inicio + longitud] for inicio in range(0, len(senal) - longitud + 1, salto)]
    return tramas or [senal[:longitud]]


def _espectrograma_trama(trama: np.ndarray) -> np.ndarray:
    ventana = np.hamming(VENTANA_STFT_MUESTRAS)
    inicios = np.arange(0, len(trama) - VENTANA_STFT_MUESTRAS + 1, SALTO_STFT_MUESTRAS)
    segmentos = np.stack([trama[i : i + VENTANA_STFT_MUESTRAS] for i in inicios], axis=0)
    segmentos = segmentos * ventana[None, :]
    espectro = np.fft.rfft(segmentos, n=PUNTOS_FFT, axis=1)
    return np.maximum(np.abs(espectro).T, EPS).astype(np.float64)


def _stft_audio(id_publico: int) -> np.ndarray:
    senal, frecuencia = leer_wav_normalizado(id_publico)
    if frecuencia != FRECUENCIA_OBJETIVO_HZ:
        raise ValueError(f"Se esperaban {FRECUENCIA_OBJETIVO_HZ} Hz y el audio tiene {frecuencia} Hz")
    espectrogramas = [_espectrograma_trama(trama) for trama in _dividir_en_tramas(senal)]
    spec = np.mean(np.stack(espectrogramas, axis=0), axis=0)
    return spec / max(float(np.sum(spec)), EPS)


def _hz_a_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_a_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


@lru_cache(maxsize=1)
def _banco_mel() -> np.ndarray:
    bins_frecuencia = PUNTOS_FFT // 2 + 1
    frecuencias = np.linspace(0.0, FRECUENCIA_OBJETIVO_HZ / 2.0, bins_frecuencia)
    puntos_mel = np.linspace(_hz_a_mel(0.0), _hz_a_mel(FRECUENCIA_OBJETIVO_HZ / 2.0), BANDAS_MEL + 2)
    puntos_hz = _mel_a_hz(puntos_mel)
    banco = np.zeros((BANDAS_MEL, bins_frecuencia), dtype=np.float64)
    for banda in range(BANDAS_MEL):
        izquierda, centro, derecha = puntos_hz[banda], puntos_hz[banda + 1], puntos_hz[banda + 2]
        subida = (frecuencias - izquierda) / max(centro - izquierda, EPS)
        bajada = (derecha - frecuencias) / max(derecha - centro, EPS)
        banco[banda] = np.maximum(0.0, np.minimum(subida, bajada))
    return banco


def _normalizar_columnas_w(w: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    escala = np.maximum(np.linalg.norm(w, axis=0), EPS)
    return w / escala[None, :], h * escala[:, None]


def _factorizar_onmf(matriz: np.ndarray, rango: int, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(matriz.astype(np.float64, copy=False), EPS)
    rng = np.random.default_rng(semilla)
    w = rng.random((x.shape[0], rango)) + EPS
    h = rng.random((rango, x.shape[1])) + EPS
    w, h = _normalizar_columnas_w(w, h)
    for _ in range(ITERACIONES_ONMF):
        w *= (x @ h.T) / (w @ (h @ h.T) + EPS)
        w = np.maximum(w, EPS)
        h *= (w.T @ x + PENALIZACION_ORTOGONAL * h) / (
            (w.T @ w) @ h + PENALIZACION_ORTOGONAL * ((h @ h.T) @ h) + EPS
        )
        h = np.maximum(h, EPS)
        w, h = _normalizar_columnas_w(w, h)
    return w, h


def _deep_onmf(matriz: np.ndarray, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    entrada = np.maximum(matriz.astype(np.float64, copy=False), EPS)
    matrices_w: list[np.ndarray] = []
    for indice, rango in enumerate(RANGOS_DEEP_ONMF, start=1):
        w, h = _factorizar_onmf(entrada, rango=rango, semilla=semilla + indice * 1000)
        matrices_w.append(w)
        entrada = h
    w_final = matrices_w[0] @ matrices_w[1] @ matrices_w[2]
    normas = np.maximum(np.linalg.norm(w_final, axis=0), EPS)
    return w_final / normas[None, :], entrada * normas[:, None]


@lru_cache(maxsize=16)
def _representaciones_audio_cache(id_publico: int) -> tuple[tuple[str, np.ndarray], ...]:
    indice = resolver_indice(id_publico)
    stft = _stft_audio(id_publico)
    mel = np.maximum(_banco_mel() @ stft, EPS)
    logmel = np.log(mel)
    mfcc = dct(logmel, type=2, axis=0, norm="ortho")[:COEFICIENTES_MFCC]
    w_final, h3 = _deep_onmf(stft, semilla=SEMILLA_ANONIMA + (indice + 1) * 37)
    return (
        ("STFT", stft.astype(np.float32)),
        ("MFCC", mfcc.astype(np.float32)),
        ("MelSpectrogram", mel.astype(np.float32)),
        ("LogMelSpectrogram", logmel.astype(np.float32)),
        ("DeepONMF_W", w_final.astype(np.float32)),
        ("DeepONMF_H3", h3.astype(np.float32)),
    )


def representaciones_audio(id_publico: int) -> dict[str, np.ndarray]:
    return dict(_representaciones_audio_cache(id_publico))


@lru_cache(maxsize=len(REPRESENTACIONES))
def predicciones_ujanet(representacion: str) -> pd.DataFrame:
    if representacion not in REPRESENTACIONES:
        raise ValueError(f"Representacion no soportada para UjaNet: {representacion}")
    frames = [pd.read_csv(ruta) for ruta in sorted((PREDICCIONES_DIR / representacion).glob("fold_*_predicciones.csv"))]
    if not frames:
        raise FileNotFoundError(f"No se encontraron predicciones en {PREDICCIONES_DIR / representacion}")
    return pd.concat(frames, ignore_index=True)


def _resultado_representacion(representacion: str, archivo: str, clase_real: str) -> ResultadoPrueba:
    predicciones = predicciones_ujanet(representacion)
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
    archivo = str(fila["archivo"])
    clase_real = str(fila["clase"])
    resultados = tuple(
        _resultado_representacion(representacion, archivo, clase_real)
        for representacion in REPRESENTACIONES
    )
    return Diagnostico(
        etiqueta_audio=etiqueta_audio(id_publico),
        resultados=resultados,
        clase_real=clase_real,
        clase_real_nombre=NOMBRES_CLASES.get(clase_real, clase_real),
        etiqueta_real_binaria=str(fila["etiqueta_binaria"]),
        archivo_original=archivo,
        ruta_original=str(fila["zip_path"]),
    )


def ruta_audio(id_publico: int) -> Path:
    _reconstruir_zip_si_hace_falta()
    return AUDIO_ZIP
