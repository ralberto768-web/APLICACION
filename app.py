from __future__ import annotations

import os
from io import BytesIO

from flask import Flask, abort, redirect, render_template, request, send_file, url_for

from graficas import paquete_graficas
from servicios_datos import (
    bytes_audio,
    comprobar_estructura,
    diagnostico_audio,
    etiqueta_audio,
    listar_audios,
)


app = Flask(__name__)


@app.errorhandler(Exception)
def manejar_error(error):
    codigo = getattr(error, "code", 500)
    return render_template("error.html", error=error), codigo


@app.route("/", methods=["GET"])
def inicio():
    comprobar_estructura()
    seleccionado = request.args.get("audio", "1")
    try:
        seleccionado_int = int(seleccionado)
        etiqueta_audio(seleccionado_int)
    except Exception:
        seleccionado_int = 1
    return render_template(
        "index.html",
        pantalla="inicio",
        audios=listar_audios(),
        seleccionado=seleccionado_int,
    )


@app.route("/accion", methods=["POST"])
def accion():
    audio_id = int(request.form.get("audio_id", "1"))
    accion_solicitada = request.form.get("accion", "representar")
    if accion_solicitada == "diagnosticar":
        return redirect(url_for("diagnosticar", audio_id=audio_id))
    return redirect(url_for("representar", audio_id=audio_id))


@app.route("/representar/<int:audio_id>", methods=["GET"])
def representar(audio_id: int):
    etiqueta = etiqueta_audio(audio_id)
    graficas = paquete_graficas(audio_id)
    return render_template(
        "index.html",
        pantalla="representar",
        audio_id=audio_id,
        etiqueta=etiqueta,
        graficas=graficas,
    )


@app.route("/diagnosticar/<int:audio_id>", methods=["GET"])
def diagnosticar(audio_id: int):
    validar = request.args.get("validar") == "1"
    diagnostico = diagnostico_audio(audio_id)
    return render_template(
        "index.html",
        pantalla="diagnosticar",
        audio_id=audio_id,
        diagnostico=diagnostico,
        validar=validar,
    )


@app.route("/audio/<int:audio_id>", methods=["GET"])
def audio(audio_id: int):
    try:
        contenido = bytes_audio(audio_id)
    except Exception:
        abort(404)
    return send_file(BytesIO(contenido), mimetype="audio/wav", download_name=f"audio{audio_id}.wav")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
