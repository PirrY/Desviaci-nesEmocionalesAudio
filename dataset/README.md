# Dataset — RAVDESS para Simulación de Llamadas de Call Center

## Fuente

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Audio
- 24 actores profesionales (12 hombres, 12 mujeres)
- 2 frases estándar grabadas en 8 emociones × 2 intensidades
- Formato: `.wav`, 48 kHz, estéreo

```python
import kagglehub
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
```

O ejecutar directamente:
```bash
python scripts/download_ravdess.py
```

---

## Convención de nombres RAVDESS

Cada archivo sigue el esquema:
```
03-01-06-01-02-01-12.wav
│  │  │  │  │  │  └── Actor (01–24)
│  │  │  │  │  └────── Repetición (01=1ª, 02=2ª)
│  │  │  │  └───────── Enunciado (01="Kids are talking...", 02="Dogs are sitting...")
│  │  │  └──────────── Intensidad (01=normal, 02=fuerte)
│  │  └─────────────── Emoción (ver tabla)
│  └────────────────── Canal vocal (01=habla, 02=canción)
└───────────────────── Modalidad (03=solo audio)
```

### Mapa de emociones

| Código | Emoción | Relevancia para call center |
|--------|---------|----------------------------|
| 01 | neutral | Línea base (estado de referencia) |
| 02 | calm | Línea base alternativa |
| 03 | happy | Activación positiva / cliente satisfecho |
| 04 | sad | Baja activación / desánimo |
| 05 | angry | **Pico crítico — escalamiento** |
| 06 | fearful | **Pico crítico — estrés / ansiedad** |
| 07 | disgust | Rechazo / insatisfacción fuerte |
| 08 | surprised | Activación súbita |

---

## Resumen del dataset actual

| Actor | Género | Llamadas | Duración total |
|-------|--------|----------|---------------|
| 01 | Hombre | 10 | 21.8 min |
| 02 | Mujer  | 10 | 23.3 min |
| 03 | Hombre | 10 | 22.7 min |
| 04 | Mujer  | 10 | 21.9 min |
| **Total** | | **40** | **~90 min** |

- Duración promedio por llamada: ~135 s
- Segmentos totales: 1232 (180 de pico = 14.6% del total)
- Segmentos de pico siempre anotados con `valence=high_activation`

### Distribución de arquetipos

| Arquetipo | Llamadas | Descripción |
|-----------|----------|-------------|
| `single_peak` | 24 (60%) | Un pico en posición aleatoria (inicio / mitad / final) |
| `double_peak` | 8 (20%) | Dos picos separados por retorno a neutro |
| `flat` | 4 (10%) | Sin picos — control para medir falsos positivos |
| `escalation` | 4 (10%) | Tensión creciente sin resolución |

> La estructura de cada llamada es completamente aleatoria: arquetipo, posición de picos,
> emociones usadas, número de repeticiones y duración de silencios varían por semilla.
> Esto evita que el modelo aprenda patrones posicionales fijos.

---

## Estrategia de simulación de llamadas

### Objetivo

Simular una llamada de call center donde **un mismo hablante** (mismo actor RAVDESS) repite las mismas frases con distintas emocionalidades a lo largo del tiempo. Esto permite:

1. Tener **ground truth exacto** de cuándo ocurre cada emoción (timestamps conocidos)
2. Probar que el sistema detecta las desviaciones respecto a la parte neutral
3. Evaluar falsos positivos con las llamadas `flat` (sin picos)

### Arquetipos de llamada

Cada llamada se construye a partir de uno de cinco arquetipos elegidos aleatoriamente:

```
flat:          [neutral ──────────────────────────────────────]  sin picos
single_peak:   [neutral ── low ── PEAK ── neutral]              posición aleatoria
double_peak:   [neutral ── PEAK ── neutral ── PEAK ── neutral]  segundo pico más largo
early_peak:    [PEAK ── low ── neutral ──────────────────────]  cliente llega enojado
escalation:    [neutral ── low ─────── PEAK ────────────────]   sin resolución
```

### Formato del archivo de anotaciones

Cada llamada simulada tiene un archivo `.csv` en `annotations/` con:

```csv
filename,start_s,end_s,emotion,emotion_code,intensity,actor,valence
sim_actor01_call01.wav,0.0,8.4,neutral,01,01,01,neutral
sim_actor01_call01.wav,8.4,17.1,neutral,01,01,01,neutral
sim_actor01_call01.wav,17.1,25.8,calm,02,01,01,neutral
sim_actor01_call01.wav,25.8,34.5,angry,05,02,01,high_activation
...
```

El campo `valence` clasifica cada segmento en:
- `neutral` — línea base
- `high_activation` — angry, fearful, surprised, happy intenso
- `low_activation` — sad, disgust

---

## Selección de actores recomendada

Para simular el rol de **asesor** (hablante consistente durante toda la llamada):
- Actor 01 (hombre) o Actor 02 (mujer) — comenzar con uno solo para minimizar variabilidad
- Usar ambas frases (`statement 01` y `statement 02`) para simular variedad léxica
- Incluir ambas intensidades (`normal` y `fuerte`) para el segmento de pico

Para simular el rol de **cliente** (el que escala emocionalmente):
- Actor diferente al asesor
- Misma lógica de selección

---

## Estructura de carpetas

```
dataset/
├── raw/                          # Archivos RAVDESS originales (no versionados)
│   └── Actor_01/ … Actor_24/
├── simulated_calls/              # Llamadas continuas generadas (.wav, no versionadas)
│   ├── sim_actor01_call01.wav
│   ├── sim_actor01_call02.wav
│   └── ...
├── annotations/                  # Ground truth (sí versionado)
│   ├── sim_actor01_call01.csv
│   └── manifest.json             # Resumen de todas las llamadas
├── scripts/
│   ├── download_ravdess.py       # Descarga el dataset desde Kaggle
│   └── build_simulated_call.py  # Genera las llamadas simuladas
└── README.md                     # Este archivo
```

---

## Reproducibilidad

El script `build_simulated_call.py` acepta una semilla aleatoria (`--seed`) para que las llamadas generadas sean 100% reproducibles. Las anotaciones en `annotations/` se generan automáticamente durante la construcción y quedan versionadas en el repositorio, aunque los `.wav` no lo estén.

Para regenerar todas las llamadas desde cero:
```bash
python scripts/build_simulated_call.py --actor 01 --n-calls 5 --seed 42 \
    --raw-dir raw/ --output-dir simulated_calls/ --annotations-dir annotations/
```
