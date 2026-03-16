# Detección de Picos Emocionales en Llamadas de Call Center
### Análisis Paralingüístico y Machine Learning

**Estudiante:** Daniel Giraldo Valencia
**Tutor:** Juan David Correa
**Programa:** Ingeniería de Sistemas

---

## Descripción

Los call centers procesan llamadas donde la evaluación de calidad requiere que supervisores escuchen grabaciones completas para identificar momentos críticos de escalamiento emocional. Este proyecto automatiza ese proceso mediante análisis de características paralingüísticas del habla — tono, ritmo, volumen y calidad vocal — para detectar picos emocionales sin depender del contenido textual.

El sistema establece una línea base emocional personalizada por hablante y detecta desviaciones significativas que indican activación emocional (enojo, excitación, estrés, ansiedad).

---

## Objetivos

**General:** Desarrollar un sistema automatizado de detección de picos emocionales en llamadas de call center basado en análisis de características paralingüísticas del habla.

**Específicos:**
1. Implementar pipeline de extracción de características acústicas (F0, intensidad, jitter/shimmer, MFCCs, HNR)
2. Evaluar métodos de cálculo de líneas base acústicas individualizadas
3. Implementar y comparar algoritmos de detección de anomalías (Z-scores, Isolation Forest, One-Class SVM)
4. Construir clasificador de valencia emocional (alta activación vs. baja activación)
5. Validar el sistema con llamadas reales y anotaciones de expertos

**Métricas objetivo:** Precision ≥75% · Recall ≥70% · F1 ≥72% · Clasificación de valencia ≥65%

---

## Estructura del proyecto

```
.
├── POC/
│   └── notebooks/
│       ├── preprocessing.ipynb           # Paso 1 — Limpieza y normalización del audio
│       ├── emotional_baseline.ipynb      # Paso 2 — Diarización, baseline y detección
│       ├── paralinguistic_features.ipynb # Extracción y análisis de features acústicas
│       └── experiments.ipynb             # Validación con audio sintético
├── dataset/                              # Dataset RAVDESS + llamadas simuladas
│   ├── raw/                              # Archivos RAVDESS originales (no versionados)
│   ├── simulated_calls/                  # Llamadas continuas generadas (no versionadas)
│   ├── annotations/                      # Ground truth: timestamps y distribución de emociones
│   ├── scripts/                          # Scripts de descarga y construcción de llamadas
│   └── README.md
├── entregables/
│   ├── entrega1/                         # Entrega1.odt
│   ├── entrega2/                         # Entrega2.odt · Entrega2.pdf
│   └── entrega3/                         # Entrega3.odt · Entrega3.pdf
├── output_preprocessing/                 # Audio limpio + diagnósticos
├── output_emotional/                     # Resultados del análisis emocional
└── output_experiments/                   # Resultados del experimento controlado
```

---

## Dataset

Se utiliza **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Audio) — corpus actuado por 24 actores profesionales que graban enunciados en 8 categorías emocionales. Es el benchmark estándar para reconocimiento de emociones en voz.

Para simular llamadas de call center, se toman grabaciones de **un único actor** (mismo hablante, frases idénticas, distintas emociones) y se concatenan en una grabación continua con una distribución de emociones conocida. Esto permite validación controlada con ground truth exacto.

```bash
# Descargar dataset
pip install kagglehub
python dataset/scripts/download_ravdess.py
```

Ver [`dataset/README.md`](dataset/README.md) para la metodología completa de simulación y la distribución de emociones.

---

## Flujo completo

```
Audio de llamada (.wav / .mp3 / .m4a)
         │
         ▼
┌─────────────────────┐
│  preprocessing.ipynb │   Limpia el audio (ruido, clipping, normalización)
└─────────┬───────────┘
          │  llamada_clean.wav
          ▼
┌──────────────────────────┐
│  emotional_baseline.ipynb │   Separa hablantes y detecta picos emocionales
└──────────────────────────┘
          │
          ▼
   Picos detectados + arco emocional + CSVs con timestamps
```

Para validar el sistema sin audio real, `experiments.ipynb` genera audio sintético con picos inyectados en posiciones conocidas y mide precision/recall.

---

## Notebooks

### `preprocessing.ipynb` — Preprocesamiento de audio

Pipeline de 8 pasos para maximizar la calidad de las estimaciones posteriores:

| Paso | Técnica | Problema que resuelve |
|------|---------|----------------------|
| 1 | Conversión mono / 16 kHz | Formato requerido por pyannote y librosa |
| 2 | Eliminación de DC offset | Desplazamiento de continua que infla el RMS y distorsiona el ZCR |
| 3 | Reparación de clipping | Saturación del A/D que genera armónicos falsos en jitter y shimmer |
| 4 | Reducción de ruido (spectral gating) | Ruido de fondo que desestabiliza el pitch y eleva el jitter artificialmente |
| 5 | Filtro bandpass 300–3400 Hz | Elimina zumbidos (< 300 Hz) y ruido de alta frecuencia (> 3400 Hz) |
| 6 | Pre-énfasis (α = 0.97) | Compensa la caída natural de ~6 dB/oct de la voz |
| 7 | Normalización de loudness (LUFS) | Iguala el volumen percibido entre hablantes |
| 8 | VAD (detección de actividad vocal) | Recorta silencios > 0.5 s para no contaminar la línea base |

**Salidas:** `llamada_clean.wav`, `llamada_clean_vad.wav`, `preprocessing_metadata.json`, gráficos de diagnóstico.

**Parámetros ajustables:**
```python
BP_LOW               = 300     # Hz — corte inferior del bandpass
BP_HIGH              = 3400    # Hz — corte superior (8000 Hz para audio HD)
PREEMPH_COEF         = 0.97    # coeficiente de pre-énfasis (0.90–0.99)
TARGET_LUFS          = -16.0   # loudness objetivo (−23 LUFS = broadcast)
NOISE_REDUCTION_PROP = 1.0     # agresividad de reducción de ruido
VAD_MIN_SILENCE_S    = 0.5     # silencios más cortos que esto se conservan
```

---

### `emotional_baseline.ipynb` — Detección de picos emocionales

#### Diarización de hablantes
Usa **pyannote/speaker-diarization-3.1** para separar asesor y cliente. Requiere cuenta en HuggingFace y aceptar los términos del modelo.

#### Features paralingüísticas

Para cada ventana de 5 s (hop 1 s):

| Feature | Librería | Implicación emocional |
|---------|---------|----------------------|
| `pitch_mean` / `pitch_std` | librosa (pyin) | Sube con alta activación (ira, miedo, excitación) |
| `energy_db` | librosa | Aumenta con activación emocional y estrés |
| `speech_rate` | librosa (onset) | Se acelera con ansiedad, ralentiza con tristeza |
| `jitter` | parselmouth (Praat) | Aumenta con tensión emocional y física |
| `hnr` | parselmouth (Praat) | Baja bajo estrés (voz más ruidosa/áspera) |
| `mfccs` (13) | librosa | Capturan timbre vocal para clasificación holística |

#### Métodos de línea base (comparados)

| Método | Descripción | Cuándo usar |
|--------|-------------|-------------|
| `first_n` | Primeros 60 s de habla | Inicio de llamada asumido neutral |
| `iqr` | Mediana ± 1.5 × IQR de la llamada completa | Cuando el inicio ya es tenso |
| `sliding` | Ventana deslizante adaptativa | Llamadas largas con cambios de estado prolongados |

#### Detección de picos

Score de desviación por ventana:
```
z_i(t) = (feature_i(t) − media_base_i) / std_base_i
S(t)   = Σ w_i × |z_i(t)|
```
Umbral adaptativo por hablante: `media(S) + 2σ`. Umbral configurable via `ALERT_THRESHOLD`.

#### Clasificación de valencia

| z_pitch + z_energy | Interpretación |
|---------------------|----------------|
| Positivo (> 0) | Alta activación: enojo, excitación, nerviosismo |
| Negativo (< 0) | Baja activación: tristeza, apatía, calma extrema |

Para clasificación más robusta: SVM o Random Forest entrenado sobre z-scores (Fase 4 del proyecto).

---

### `experiments.ipynb` — Validación con audio sintético

Notebook autocontenido que genera su propio audio con eventos inyectados en posiciones conocidas y mide el rendimiento del pipeline completo.

| Experimento | Variable | Métrica |
|------------|---------|---------|
| Detección básica | — | TP / FP / FN / F1 |
| Sweep de umbral | `ALERT_THRESHOLD` 0.8–3.5 | Curva Precision-Recall |
| Duración de base | 10–60 s | F1 por duración |
| Distribuciones | — | KDE neutro vs. evento por feature |

---

## Instalación

```bash
# Audio y features
pip install librosa soundfile pydub praat-parselmouth noisereduce pyloudnorm scipy

# Diarización (requiere aceptar términos en HuggingFace)
pip install pyannote.audio

# Torch (CPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Dataset
pip install kagglehub

# Visualización y datos
pip install matplotlib seaborn pandas numpy scikit-learn
```

---

## Uso rápido

```bash
# Con audio real — editar AUDIO_PATH y HF_TOKEN en los notebooks primero
jupyter nbconvert --to notebook --execute preprocessing.ipynb --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute emotional_baseline.ipynb

# Validación sin audio real
jupyter nbconvert --to notebook --execute experiments.ipynb

# Construir llamada simulada desde RAVDESS
python dataset/scripts/build_simulated_call.py --actor 01 --output dataset/simulated_calls/
```

---

## Limitaciones conocidas

- **Jitter/Shimmer** son sensibles al ruido de canal telefónico (compresión GSM). Interpretarlos con cautela en audio de baja calidad.
- **`speech_rate` via onset detection** es una aproximación. Para mayor precisión, integrar ASR (Whisper) y contar sílabas reales.
- **La base calculada en la misma llamada** asume que la mayor parte es neutral. Si la persona está en tensión durante toda la llamada, la base queda contaminada — usar grabación de referencia externa.
- **pyannote puede confundir hablantes** en cruces de voz o habla simultánea.
- **RAVDESS es habla actuada** — el rendimiento puede caer en habla espontánea real (dominio shift conocido en la literatura).

---

## Hoja de ruta

| Fase | Semanas | Objetivo |
|------|---------|---------|
| 1 | 1–3 | Pipeline de extracción de features acústicas |
| 2 | 4–5 | Métodos de baseline emocional + preprocesamiento |
| 3 | 6–8 | Detector de anomalías (Z-scores + Isolation Forest) |
| 4 | 9–10 | Clasificador de valencia (SVM / Random Forest) |
| 5 | 11–13 | Validación con llamadas reales + análisis comparativo |
| 6 | 14–15 | Documentación final + demo |
