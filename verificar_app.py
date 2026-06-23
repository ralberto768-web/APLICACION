from __future__ import annotations

from servicios_datos import diagnostico_audio, listar_audios, representaciones_audio


def main() -> int:
    audios = listar_audios()
    assert len(audios) == 1000, f"Se esperaban 1000 audios y hay {len(audios)}"
    assert audios[0].etiqueta == "PCG 0001"
    assert audios[-1].etiqueta == "PCG 1000"

    for audio_id in (1, 250, 500, 750, 1000):
        reps = representaciones_audio(audio_id)
        assert set(reps) == {
            "STFT",
            "MFCC",
            "MelSpectrogram",
            "LogMelSpectrogram",
            "DeepONMF_W",
            "DeepONMF_H3",
        }
        diag = diagnostico_audio(audio_id)
        assert len(diag.resultados) == 6
        assert all(resultado.prediccion_multiclase for resultado in diag.resultados)
        assert diag.archivo_original.endswith(".wav")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
