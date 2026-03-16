"""
Construye llamadas de call center simuladas a partir de grabaciones RAVDESS
de un único actor. La estructura de cada llamada es completamente aleatoria:
arquetipo distinto, posición de picos variable, duración variable.

Arquetipos de llamada:
  - flat:           solo segmentos neutros/calm (control — sin picos)
  - single_peak:    un pico emocional en posición aleatoria
  - double_peak:    dos picos separados (el segundo más intenso)
  - early_peak:     pico al principio, resolución larga
  - escalation:     tensión creciente sin retorno a neutro

Genera:
  - simulated_calls/sim_actor{N}_call{K}.wav  — audio continuo
  - annotations/sim_actor{N}_call{K}.csv      — ground truth con timestamps

Uso:
    python build_simulated_call.py --actor 01 --n-calls 10 --seed 42
    python build_simulated_call.py --actor 01 --n-calls 10 --seed 42 --min-duration 120 --max-duration 240

Requiere:
    pip install soundfile numpy scipy
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ── Parámetros globales ───────────────────────────────────────────────────────
TARGET_SR       = 16_000   # Hz
SILENCE_MIN_S   = 0.3      # silencio mínimo entre segmentos
SILENCE_MAX_S   = 1.2      # silencio máximo entre segmentos
MIN_DURATION_S  = 90       # duración mínima de llamada por defecto
MAX_DURATION_S  = 210      # duración máxima de llamada por defecto

EMOTION_NAMES = {
    "01": "neutral", "02": "calm",    "03": "happy",    "04": "sad",
    "05": "angry",   "06": "fearful", "07": "disgust",  "08": "surprised",
}

# Emociones agrupadas por rol en la llamada
BASELINE_EMOTIONS  = [("01", "01"), ("02", "01")]   # neutro / calmado
LOW_ACT_EMOTIONS   = [("04", "01"), ("07", "01")]   # queja leve / disgusto
HIGH_ACT_EMOTIONS  = [("05", "01"), ("05", "02"),   # enojo normal / fuerte
                      ("06", "01"), ("06", "02"),   # miedo/estrés normal / fuerte
                      ("07", "02"), ("03", "02")]   # disgusto fuerte / alegría intensa

# Arquetipos disponibles y su peso de muestreo
ARCHETYPES = [
    ("flat",         0.15),
    ("single_peak",  0.35),
    ("double_peak",  0.25),
    ("early_peak",   0.15),
    ("escalation",   0.10),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_actor_files(raw_dir: Path, actor: str) -> list[Path]:
    actor_dir = raw_dir / f"Actor_{actor.zfill(2)}"
    if not actor_dir.exists():
        raise FileNotFoundError(
            f"No se encontró {actor_dir}. Ejecuta download_ravdess.py primero."
        )
    files = list(actor_dir.glob("03-01-*.wav"))
    if not files:
        raise ValueError(f"No hay archivos .wav en {actor_dir}")
    return files


def parse_filename(path: Path) -> dict:
    parts = path.stem.split("-")
    return {
        "modality":   parts[0], "channel":    parts[1],
        "emotion":    parts[2], "intensity":  parts[3],
        "statement":  parts[4], "repetition": parts[5],
        "actor":      parts[6],
    }


def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sr)
    return audio.astype(np.float32)


def pick(files: list[Path], emotion_code: str, intensity: str) -> Path:
    """Elige un archivo aleatorio para la emoción/intensidad dada. Permite reusar."""
    candidates = [
        f for f in files
        if parse_filename(f)["emotion"] == emotion_code
        and parse_filename(f)["intensity"] == intensity
    ]
    # Fallback: misma emoción, cualquier intensidad
    if not candidates:
        candidates = [f for f in files if parse_filename(f)["emotion"] == emotion_code]
    if not candidates:
        raise ValueError(f"Sin archivos para emoción {emotion_code}")
    return random.choice(candidates)


def pick_group(files, group: list[tuple]) -> tuple[Path, str]:
    """Elige aleatoriamente de un grupo de (emotion_code, intensity) y retorna (path, valence)."""
    emotion_code, intensity = random.choice(group)
    if group is HIGH_ACT_EMOTIONS:
        valence = "high_activation"
    elif group is LOW_ACT_EMOTIONS:
        valence = "low_activation"
    else:
        valence = "neutral"
    return pick(files, emotion_code, intensity), valence


def silence_chunk() -> np.ndarray:
    dur = random.uniform(SILENCE_MIN_S, SILENCE_MAX_S)
    return np.zeros(int(TARGET_SR * dur), dtype=np.float32)


# ── Bloques reutilizables ─────────────────────────────────────────────────────

def block_baseline(files, n_min=3, n_max=8) -> list[tuple]:
    """Bloque de segmentos neutros/calm."""
    return [pick_group(files, BASELINE_EMOTIONS) for _ in range(random.randint(n_min, n_max))]


def block_low(files, n_min=1, n_max=4) -> list[tuple]:
    """Bloque de tensión baja (queja, desánimo)."""
    return [pick_group(files, LOW_ACT_EMOTIONS) for _ in range(random.randint(n_min, n_max))]


def block_peak(files, n_min=2, n_max=6) -> list[tuple]:
    """Bloque de pico emocional alto (enojo, estrés)."""
    # Fijamos la emoción del pico para que sea coherente dentro del bloque
    emotion_code, intensity = random.choice(HIGH_ACT_EMOTIONS)
    valence = "high_activation"
    segments = []
    for _ in range(random.randint(n_min, n_max)):
        # Variamos intensidad pero mantenemos la misma emoción base
        alt_intensity = random.choice(["01", "02"])
        p = pick(files, emotion_code, alt_intensity)
        segments.append((p, valence))
    return segments


# ── Arquetipos ────────────────────────────────────────────────────────────────

def archetype_flat(files) -> list[tuple]:
    """Solo segmentos neutrales. Llamada sin picos — útil para FP testing."""
    return block_baseline(files, n_min=10, n_max=25)


def archetype_single_peak(files) -> list[tuple]:
    """Un único pico en posición aleatoria: antes, a la mitad o al final."""
    position = random.choice(["early", "mid", "late"])
    pre_neutral  = block_baseline(files, 3, 7)
    low_ramp     = block_low(files, 1, 3)
    peak         = block_peak(files, 2, 5)
    post_neutral = block_baseline(files, 2, 6)

    if position == "early":
        return peak + block_baseline(files, 5, 12)
    elif position == "mid":
        return pre_neutral + low_ramp + peak + post_neutral
    else:  # late
        return block_baseline(files, 5, 12) + low_ramp + peak


def archetype_double_peak(files) -> list[tuple]:
    """Dos picos separados por retorno a neutro. El segundo puede ser más intenso."""
    intro       = block_baseline(files, 3, 6)
    ramp1       = block_low(files, 1, 2)
    peak1       = block_peak(files, 2, 4)
    middle      = block_baseline(files, 2, 5)
    ramp2       = block_low(files, 1, 3)
    peak2       = block_peak(files, 3, 6)   # segundo pico más largo
    outro       = block_baseline(files, 1, 4)
    return intro + ramp1 + peak1 + middle + ramp2 + peak2 + outro


def archetype_early_peak(files) -> list[tuple]:
    """Cliente llega enojado, se calma durante la llamada."""
    peak         = block_peak(files, 3, 6)
    cool_down    = block_low(files, 2, 4)
    resolution   = block_baseline(files, 5, 14)
    return peak + cool_down + resolution


def archetype_escalation(files) -> list[tuple]:
    """Tensión creciente sin resolución. Nunca vuelve a neutro."""
    intro   = block_baseline(files, 3, 6)
    ramp    = block_low(files, 3, 6)
    peak    = block_peak(files, 4, 8)
    return intro + ramp + peak


ARCHETYPE_FNS = {
    "flat":         archetype_flat,
    "single_peak":  archetype_single_peak,
    "double_peak":  archetype_double_peak,
    "early_peak":   archetype_early_peak,
    "escalation":   archetype_escalation,
}


def sample_archetype() -> str:
    names, weights = zip(*ARCHETYPES)
    return random.choices(names, weights=weights, k=1)[0]


# ── Construcción de llamada ───────────────────────────────────────────────────

def build_call(raw_dir: Path, actor: str, call_idx: int, seed: int,
               output_dir: Path, annotations_dir: Path,
               min_duration: int, max_duration: int) -> dict:
    random.seed(seed + call_idx * 97)   # separación entre calls
    np.random.seed(seed + call_idx * 97)

    files = list_actor_files(raw_dir, actor)
    archetype = sample_archetype()

    # Generar segmentos según arquetipo; extender si la llamada es muy corta
    segments: list[tuple] = ARCHETYPE_FNS[archetype](files)

    # Estimar duración (aprox 3.5 s por segmento + silencio)
    def estimate_dur(segs):
        return len(segs) * (3.5 + (SILENCE_MIN_S + SILENCE_MAX_S) / 2)

    while estimate_dur(segments) < min_duration:
        # Extender añadiendo un bloque baseline al final
        segments += block_baseline(files, 3, 6)

    # Si es muy larga, truncar manteniendo el arquetipo (no cortar a la mitad de un pico)
    if estimate_dur(segments) > max_duration:
        target_n = int(max_duration / (3.5 + (SILENCE_MIN_S + SILENCE_MAX_S) / 2))
        segments = segments[:target_n]

    # Cargar audio y concatenar
    chunks, timestamps = [], []
    cursor = 0.0

    for path, valence in segments:
        meta = parse_filename(path)
        audio = load_wav(path)
        start = cursor
        end   = cursor + len(audio) / TARGET_SR
        timestamps.append({
            **meta,
            "valence":  valence,
            "start_s":  round(start, 3),
            "end_s":    round(end, 3),
        })
        chunks.append(audio)
        chunks.append(silence_chunk())
        cursor = end + (SILENCE_MIN_S + SILENCE_MAX_S) / 2  # aprox para cursor

    full_audio = np.concatenate(chunks)
    duration_s = len(full_audio) / TARGET_SR

    # ── Guardar WAV ───────────────────────────────────────────────────────────
    actor_padded = actor.zfill(2)
    out_name     = f"sim_actor{actor_padded}_call{call_idx:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{out_name}.wav"
    sf.write(str(wav_path), full_audio, TARGET_SR)

    # ── Guardar CSV de anotaciones ────────────────────────────────────────────
    annotations_dir.mkdir(parents=True, exist_ok=True)
    csv_path   = annotations_dir / f"{out_name}.csv"
    fieldnames = ["filename", "start_s", "end_s", "emotion", "emotion_code",
                  "intensity", "actor", "valence"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in timestamps:
            writer.writerow({
                "filename":     f"{out_name}.wav",
                "start_s":      t["start_s"],
                "end_s":        t["end_s"],
                "emotion":      EMOTION_NAMES[t["emotion"]],
                "emotion_code": t["emotion"],
                "intensity":    t["intensity"],
                "actor":        t["actor"],
                "valence":      t["valence"],
            })

    peak_segs = sum(1 for t in timestamps if t["valence"] == "high_activation")
    print(f"  [{archetype:<14}]  {out_name}.wav  "
          f"{duration_s:.0f}s  {len(segments)} segs  peaks={peak_segs}")

    return {
        "filename":   f"{out_name}.wav",
        "actor":      actor_padded,
        "call_idx":   call_idx,
        "seed":       seed + call_idx * 97,
        "archetype":  archetype,
        "duration_s": round(duration_s, 2),
        "n_segments": len(segments),
        "n_peaks":    peak_segs,
        "annotations": f"{out_name}.csv",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Construye llamadas simuladas desde RAVDESS")
    parser.add_argument("--actor",           default="01")
    parser.add_argument("--n-calls",         type=int, default=10)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--min-duration",    type=int, default=MIN_DURATION_S)
    parser.add_argument("--max-duration",    type=int, default=MAX_DURATION_S)
    parser.add_argument("--raw-dir",         default=None)
    parser.add_argument("--output-dir",      default=None)
    parser.add_argument("--annotations-dir", default=None)
    args = parser.parse_args()

    base            = Path(__file__).parent.parent
    raw_dir         = Path(args.raw_dir)         if args.raw_dir         else base / "raw"
    output_dir      = Path(args.output_dir)      if args.output_dir      else base / "simulated_calls"
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else base / "annotations"

    print(f"Actor: {args.actor.zfill(2)}  |  Llamadas: {args.n_calls}  |  "
          f"Duración: {args.min_duration}–{args.max_duration}s  |  Seed: {args.seed}\n")

    manifest = []
    for i in range(1, args.n_calls + 1):
        entry = build_call(raw_dir, args.actor, i, args.seed,
                           output_dir, annotations_dir,
                           args.min_duration, args.max_duration)
        manifest.append(entry)

    # Actualizar manifest.json (sobrescribe entradas del mismo actor/call_idx)
    manifest_path = annotations_dir / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    new_names = {e["filename"] for e in manifest}
    merged = [e for e in existing if e["filename"] not in new_names] + manifest
    merged.sort(key=lambda e: e["filename"])
    manifest_path.write_text(json.dumps(merged, indent=2))

    # Resumen
    archetypes_used = {}
    for e in manifest:
        archetypes_used[e["archetype"]] = archetypes_used.get(e["archetype"], 0) + 1
    total_dur = sum(e["duration_s"] for e in manifest)
    print(f"\nResumen de esta ejecución:")
    print(f"  Total:      {len(manifest)} llamadas  /  {total_dur:.0f} s  ({total_dur/60:.1f} min)")
    print(f"  Arquetipos: {archetypes_used}")
    print(f"  Manifest:   {manifest_path}  ({len(merged)} entradas totales)")


if __name__ == "__main__":
    main()
