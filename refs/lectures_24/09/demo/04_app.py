"""
Dash Speech-to-Text Demo (Harvard.wav, Vosk, Dash)

Requirements:
- dash, plotly, soundfile, numpy, vosk
- Download and extract the Vosk model (see lecture markdown for instructions)
- Place 'harvard.wav' in a 'data/' directory

Run with: python 04_app.py
"""

import soundfile as sf
import numpy as np
import plotly.graph_objs as go
import json
from vosk import Model, KaldiRecognizer
import tempfile
from dash import Dash, dcc, html
import base64
import os

# Load and process harvard.wav at startup
wav_path = os.path.join("data", "harvard.wav")
data, samplerate = sf.read(wav_path)
# Convert to mono if stereo
if data.ndim > 1:
    data = data.mean(axis=1)
max_seconds = 15
if len(data) > samplerate * max_seconds:
    data = data[: samplerate * max_seconds]

# Transcribe with Vosk
model_path = "vosk-model-small-en-us-0.15"
model = Model(model_path)
rec = KaldiRecognizer(model, samplerate)
rec.SetWords(True)
with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
    sf.write(tmp.name, data, samplerate, subtype="PCM_16")
    with sf.SoundFile(tmp.name) as f:
        while True:
            buf = f.read(4000, dtype="int16")
            if len(buf) == 0:
                break
            rec.AcceptWaveform(buf.tobytes())
        result = rec.FinalResult()
result_json = json.loads(result)
words = result_json.get("result", [])
transcript = result_json.get("text", "")

# Plotly figure with word highlights and staggered word labels
# -----------------------------------------------------------
duration = len(data) / samplerate
time = np.linspace(0, duration, len(data))
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=data, mode="lines", name="Waveform"))
for w in words:
    fig.add_vrect(
        x0=w["start"],
        x1=w["end"],
        fillcolor="orange",
        opacity=0.3,
        line_width=0,
    )
for i, w in enumerate(words):
    midpoint = (w["start"] + w["end"]) / 2
    y_pos = 1.05 * np.max(data) if i % 2 == 0 else 1.15 * np.max(data)
    fig.add_annotation(
        x=midpoint,
        y=y_pos,
        text=w["word"],
        showarrow=False,
        font=dict(size=12, color="black"),
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.7)",
    )
fig.update_layout(
    title="Harvard.wav Waveform with Word Highlights",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    height=350,
    margin=dict(l=10, r=10, t=30, b=10),
)


def wavfile_to_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return "data:audio/wav;base64," + base64.b64encode(data).decode()


audio_src = wavfile_to_base64(wav_path)

app = Dash(__name__)
app.layout = html.Div([
    html.H2("Speech-to-Text Demo (Harvard.wav, Vosk, Dash)"),
    html.Audio(src=audio_src, controls=True, style={"width": "100%"}),
    dcc.Graph(figure=fig),
    html.H4("Recognized Text:"),
    html.Pre(transcript, style={"background": "#f4f4f4", "padding": "1em"}),
])

if __name__ == "__main__":
    app.run(debug=True)
