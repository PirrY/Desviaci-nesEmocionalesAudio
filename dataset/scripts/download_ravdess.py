"""
Descarga el dataset RAVDESS desde Kaggle usando kagglehub
y lo mueve a dataset/raw/.
"""

import kagglehub
import shutil
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "raw"


def download():
    print("Descargando RAVDESS desde Kaggle...")
    path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    src = Path(path)
    print(f"Descargado en: {src}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for actor_dir in src.glob("Actor_*"):
        dest = RAW_DIR / actor_dir.name
        if dest.exists():
            print(f"  Ya existe: {dest.name} — omitiendo")
            continue
        shutil.copytree(actor_dir, dest)
        print(f"  Copiado: {actor_dir.name}")

    print(f"\nDataset disponible en: {RAW_DIR.resolve()}")
    actors = sorted(RAW_DIR.glob("Actor_*"))
    print(f"Actores encontrados: {len(actors)}")
    total = sum(1 for _ in RAW_DIR.glob("Actor_*/*.wav"))
    print(f"Archivos .wav totales: {total}")


if __name__ == "__main__":
    download()
