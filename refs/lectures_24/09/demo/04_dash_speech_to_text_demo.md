# Demo 4: Dash Speech-to-Text App with Harvard.wav

> **Note:** This Dash app always loads and processes the classic public domain `harvard.wav` sample. It visualizes the waveform, highlights recognized words, plays the audio, and shows the transcript. No uploads, no buttons—just a clean, reproducible Dash app for all students.



## 1. Install Requirements

Add these to your `requirements.txt` or install directly:

```
pip install dash plotly soundfile numpy vosk
```

You also need to download a small Vosk model (see next cell).



## 2. Download a Small Vosk Model (Automated)

You can run this cell to automatically download and extract the small English Vosk model. (Requires the `requests` package.)

```python
import os
import zipfile
import requests

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_DIR = "vosk-model-small-en-us-0.15"
MODEL_ZIP = MODEL_DIR + ".zip"

def download_and_extract_vosk_model():
    if not os.path.exists(MODEL_DIR):
        print("Downloading Vosk model...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_ZIP, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(MODEL_ZIP)
        print("Model downloaded and extracted.")
    else:
        print("Vosk model already present.")

download_and_extract_vosk_model()
```



## 3. Dash App: Visualize and Transcribe Harvard.wav

```python
import soundfile as sf
import numpy as np
import plotly.graph_objs as go
import json
from vosk import Model, KaldiRecognizer
import tempfile
from dash import Dash, dcc, html
import base64

# Load and process harvard.wav at startup
wav_path = "data/harvard.wav"
data, samplerate = sf.read(wav_path)
# Convert to mono if stereo
if data.ndim > 1:
    data = data.mean(axis=1)
max_seconds = 15
if len(data) > samplerate * max_seconds:
    data = data[:samplerate * max_seconds]

# Transcribe with Vosk
model_path = "vosk-model-small-en-us-0.15"
model = Model(model_path)
rec = KaldiRecognizer(model, samplerate)
rec.SetWords(True)
with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
    sf.write(tmp.name, data, samplerate, subtype='PCM_16')
    with sf.SoundFile(tmp.name) as f:
        while True:
            buf = f.read(4000, dtype='int16')
            if len(buf) == 0:
                break
            rec.AcceptWaveform(buf.tobytes())
        result = rec.FinalResult()
result_json = json.loads(result)
words = result_json.get('result', [])
transcript = result_json.get('text', '')

# Plotly figure with word highlights and word labels above waveform
# --------------------------------------------------------------
duration = len(data) / samplerate
# Time axis for waveform
time = np.linspace(0, duration, len(data))
fig = go.Figure()
# Plot the waveform
fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name='Waveform'))
# Add shaded rectangles for each word
for w in words:
    fig.add_vrect(
        x0=w['start'], x1=w['end'],
        fillcolor='orange', opacity=0.3, line_width=0,
    )
# Add word labels as annotations above the waveform, staggered to avoid overlap
for i, w in enumerate(words):
    midpoint = (w['start'] + w['end']) / 2
    # Alternate y position for each word annotation
    y_pos = 1.05 * np.max(data) if i % 2 == 0 else 1.15 * np.max(data)
    fig.add_annotation(
        x=midpoint, y=y_pos,
        text=w['word'],
        showarrow=False,
        font=dict(size=12, color='black'),
        yanchor='bottom',
        bgcolor='rgba(255,255,255,0.7)'
    )
fig.update_layout(
    title="Harvard.wav Waveform with Word Highlights",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    height=350,
    margin=dict(l=10, r=10, t=30, b=10)
)

# Encode audio for Dash player
# ----------------------------
def wavfile_to_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return "data:audio/wav;base64," + base64.b64encode(data).decode()
audio_src = wavfile_to_base64(wav_path)

# Dash app layout
# ---------------
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
```
