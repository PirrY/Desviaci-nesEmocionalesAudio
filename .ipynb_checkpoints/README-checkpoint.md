# Detección de Desviaciones Emocionales en Llamadas

Sistema de análisis paralingüístico de llamadas telefónicas que separa hablantes,
establece una línea base de cómo habla cada persona y detecta momentos de activación
emocional (enojo, excitación, estrés) midiendo desviaciones de esa base.

---

## Estructura del proyecto

```
.
├── preprocessing.ipynb          # Paso 1 — Limpieza y normalización del audio
├── emotional_baseline.ipynb     # Paso 2 — Diarización, baseline y detección
├── experiments.ipynb            # Validación con audio sintético (sin dependencias externas)
├── output_preprocessing/        # Audio limpio + diagnósticos
├── output_emotional/            # Resultados del análisis emocional
└── output_experiments/          # Resultados del experimento controlado
```

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
│  emotional_baseline.ipynb │   Separa hablantes y detecta eventos emocionales
└──────────────────────────┘
          │
          ▼
   Eventos detectados + arco emocional + CSVs
```

Para validar el sistema sin audio real, usa `experiments.ipynb` que genera
su propio audio sintético con eventos inyectados y mide precision/recall.

---

## Notebooks

### `preprocessing.ipynb` — Preprocesamiento de audio

Prepara el audio para maximizar la calidad de las estimaciones posteriores.
Aplica un pipeline de 7 pasos en este orden:

| Paso | Técnica | Problema que resuelve |
|------|---------|----------------------|
| 1 | Conversión mono / 16 kHz | Formato requerido por pyannote y librosa |
| 2 | Eliminación de DC offset | Desplazamiento de continua que infla el RMS y distorsiona el ZCR |
| 3 | Reparación de clipping | Saturación del A/D que genera armónicos falsos en jitter y shimmer |
| 4 | Reducción de ruido (spectral gating) | Ruido de fondo que desestabiliza el pitch y eleva el jitter artificialmente |
| 5 | Filtro bandpass 300–3400 Hz | Elimina zumbidos (< 300 Hz) y ruido de alta frecuencia (> 3400 Hz) |
| 6 | Pre-énfasis (α = 0.97) | Compensa la caída natural de ~6 dB/oct de la voz, mejora estimación de pitch |
| 7 | Normalización de loudness (LUFS) | Iguala el volumen percibido entre hablantes para que la energía refleje cambios dinámicos, no niveles absolutos |
| 8 | VAD (detección de actividad vocal) | Recorta silencios > 0.5 s para no contaminar la línea base con frames vacíos |

**Salidas:**
- `output_preprocessing/llamada_clean.wav` — audio procesado, timeline original preservado (usar con diarización)
- `output_preprocessing/llamada_clean_vad.wav` — ídem pero con silencios recortados (usar para extracción de features)
- `output_preprocessing/preprocessing_metadata.json` — parámetros usados
- Gráficos de diagnóstico: `diag_raw.png`, `diag_clean.png`, `before_after.png`, `pitch_comparison.png`

**Parámetros ajustables en la celda de configuración:**

```python
BP_LOW              = 300      # Hz — corte inferior del bandpass
BP_HIGH             = 3400     # Hz — corte superior (8000 Hz para audio HD)
PREEMPH_COEF        = 0.97     # coeficiente de pre-énfasis (0.90–0.99)
TARGET_LUFS         = -16.0    # loudness objetivo (−23 LUFS = broadcast)
NOISE_REDUCTION_PROP = 1.0     # agresividad de la reducción de ruido (1.0 = conservador)
VAD_MIN_SILENCE_S   = 0.5      # silencios más cortos que esto se conservan
```

---

### `emotional_baseline.ipynb` — Análisis emocional

Dado el audio limpio (salida del paso anterior), detecta momentos de activación emocional.

#### Diarización de hablantes

Usa **pyannote/speaker-diarization-3.1** para identificar quién habla en cada momento.
Fijamos `num_speakers=2` ya que las llamadas son entre dos personas.

> Requiere: cuenta en HuggingFace + aceptar los términos del modelo en su página

#### Features paralingüísticas

Para cada ventana de análisis de 5 s (hop 1 s) se calculan:

| Feature | Librería | Qué mide |
|---------|---------|----------|
| `pitch_mean` / `pitch_std` | librosa (pyin) | Tono fundamental y su variación — sube en emociones de alta activación |
| `energy_db` | librosa | Volumen en dB — más energía = más intensidad emocional |
| `speech_rate` | librosa (onset) | Sílabas por segundo — habla más rápida en excitación/nerviosismo |
| `jitter` | Praat (parselmouth) | Variación ciclo a ciclo del pitch — voz tensa tiene más jitter |
| `hnr` | Praat (parselmouth) | Relación armónicos/ruido — baja bajo estrés (voz más ruidosa) |

#### Línea base

Dos estrategias configurables:

- **`first_n`** (default): usa los primeros 60 s de habla de cada persona. Asume inicio neutral.
- **`iqr`**: usa mediana ± 1.5 × IQR de toda la llamada. Más robusto si el inicio ya es tenso.

> Para producción, la base ideal es una grabación de referencia de esa persona en contexto neutro.

#### Score de desviación

Para cada ventana y cada feature:

```
z_i(t) = (feature_i(t) − media_base_i) / std_base_i
```

Score compuesto ponderado:

```
S(t) = Σ w_i × |z_i(t)|
```

Un `S(t)` alto indica que múltiples features se alejaron simultáneamente de la base.
El umbral de alerta es adaptativo por hablante: `media(S) + 2σ`.

#### Dirección emocional

| `z_pitch + z_energy` | Interpretación |
|---------------------|----------------|
| Positivo (`> 0`) | Alta activación: enojo, excitación, nerviosismo |
| Negativo (`< 0`) | Baja activación: tristeza, apatía, calma extrema |

**Salidas:**
- `output_emotional/emotional_arc.png` — arco emocional completo
- `output_emotional/zscore_heatmap_*.png` — qué feature disparó cada evento
- `output_emotional/events_detected.csv` — tabla de eventos con inicio, fin, tipo
- `output_emotional/windows_with_scores.csv` — todas las ventanas con scores

**Parámetros ajustables:**

```python
WINDOW_S             = 5.0     # duración de cada ventana de análisis (s)
HOP_S                = 1.0     # paso entre ventanas (s)
BASELINE_STRATEGY    = 'first_n'
BASELINE_FIRST_N_S   = 60      # segundos de habla para la base
ALERT_THRESHOLD      = 2.0     # desviaciones estándar para declarar evento
FEATURE_WEIGHTS      = {...}   # peso de cada feature en el score compuesto
```

---

### `experiments.ipynb` — Validación con audio sintético

Notebook autocontenido para validar y ajustar el sistema **sin necesitar audio real**.

#### Qué hace

1. **Genera audio sintético** con dos hablantes con F0 distintos (130 Hz y 210 Hz)
   usando un modelo de voz simplificado (suma de armónicos + modulación de amplitud)
2. **Inyecta eventos emocionales** en posiciones y duraciones conocidas (ground truth)
3. **Corre el pipeline completo** de features + baseline + detección
4. **Mide el rendimiento** contra el ground truth con IoU, precision, recall y F1

#### Experimentos incluidos

| Experimento | Variable | Métrica |
|------------|---------|---------|
| Detección básica | — | TP / FP / FN / F1 |
| Sweep de umbral | `ALERT_THRESHOLD_SD` de 0.8 a 3.5 | Curva Precision-Recall |
| Duración de base | 10 s / 20 s / 30 s / 45 s / 60 s | F1 por duración |
| Distribuciones | — | KDE neutro vs. evento por feature |

#### Eventos inyectados

```
SPEAKER_A (F0 base ~130 Hz):
  ~t=40s  → EVENTO_ENOJO_A       (pitch ×1.45, volumen ×1.80, 7s)
  ~t=48s  → EVENTO_ENOJO_A_cont  (pitch ×1.30, volumen ×1.50, 5s)

SPEAKER_B (F0 base ~210 Hz):
  ~t=80s  → EVENTO_EXCITACION_B      (pitch ×1.50, volumen ×1.70, 8s)
  ~t=93s  → EVENTO_EXCITACION_B_cont (pitch ×1.35, volumen ×1.40, 5s)
```

---

## Instalación

```bash
# Audio y features
pip install librosa soundfile pydub praat-parselmouth noisereduce pyloudnorm scipy

# Diarización (requiere aceptar términos en HuggingFace)
pip install pyannote.audio

# Torch (CPU, sin GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Visualización y datos
pip install matplotlib seaborn pandas numpy
```

---

## Uso rápido

### Con audio real

```bash
# 1. Limpiar el audio
jupyter nbconvert --to notebook --execute preprocessing.ipynb \
  --ExecutePreprocessor.timeout=600

# 2. Análisis emocional (editar AUDIO_PATH y HF_TOKEN antes)
jupyter nbconvert --to notebook --execute emotional_baseline.ipynb
```

### Validación sin audio real

```bash
jupyter nbconvert --to notebook --execute experiments.ipynb
```

---

## Limitaciones conocidas

- **Jitter/Shimmer** son sensibles al ruido de canal telefónico (compresión GSM). Interpretarlos con cautela en audio de baja calidad.
- **`speech_rate` via onset detection** es una aproximación. Para mayor precisión, integrar ASR (Whisper) y contar palabras/sílabas reales.
- **La base calculada en la misma llamada** asume que la mayor parte es neutral. Si la persona está en tensión durante toda la llamada, la base queda contaminada. La solución es usar una grabación de referencia externa.
- **pyannote puede confundir hablantes** en cruces de voz o cuando hablan simultáneamente. El pre-énfasis y la reducción de ruido del paso de preprocesamiento ayudan pero no eliminan el problema.
- La **dirección emocional** (enojo vs. tristeza) es una heurística basada en pitch + energía. Para clasificación más robusta, entrenar un modelo supervisado con las features como entrada.

---

## Próximos pasos para producción

1. **Base externa**: grabar a cada agente en una sesión neutra conocida y usar esa grabación como referencia permanente.
2. **Integrar ASR**: transcribir con Whisper y correlacionar eventos acústicos con palabras concretas.
3. **Clasificador supervisado**: etiquetar manualmente un subconjunto de llamadas y entrenar un SVM o Random Forest sobre los z-scores.
4. **Cruzar con métricas de negocio**: correlacionar eventos emocionales con resultado de la llamada (conversión, NPS, escalaciones).
5. **Streaming**: adaptar el pipeline para procesar en tiempo real con ventanas deslizantes sobre el audio en vivo.
