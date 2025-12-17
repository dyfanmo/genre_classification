from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import librosa
from panns_inference import AudioTagging

from src.config import (
    GENRES_DIR,
    EMBEDDINGS_CSV,
    SAMPLE_RATE,
    CLIP_DURATION_SECONDS,
)
from src.utils import get_device, ensure_directory


def load_panns_model() -> AudioTagging:
    device = get_device()
    print(f"Using device: {device.upper()}")
    model = AudioTagging(checkpoint_path=None, device=device)
    return model


def list_audio_files(genres_dir: Path) -> Iterable[Tuple[str, Path]]:
    for genre_dir in sorted(genres_dir.iterdir()):
        if not genre_dir.is_dir():
            continue

        genre_name = genre_dir.name
        for wav_path in sorted(genre_dir.glob("*.wav")):
            yield genre_name, wav_path


def load_audio_clip(wav_path: Path) -> Optional[np.ndarray]:
    try:
        audio, _sr = librosa.load(
            path=str(wav_path),
            sr=SAMPLE_RATE,
            mono=True,
            duration=CLIP_DURATION_SECONDS,
        )
        return audio
    except Exception as exc:
        print(f"[WARN] Skipping {wav_path} – failed to load audio: {exc}")
        return None


def embed_waveform(model: AudioTagging, audio: np.ndarray) -> np.ndarray:
    audio_batch = audio[None, :]
    _clipwise_output, embedding = model.inference(audio_batch)
    return embedding[0]


def build_embedding_row(
    genre: str,
    wav_path: Path,
    embedding: np.ndarray,
) -> Dict[str, Any]:

    row = {
        "file": f"{genre}/{wav_path.name}",
        "genre": genre,
    }
    for i, value in enumerate(embedding):
        row[f"e_{i}"] = float(value)
    return row


def generate_embeddings_dataframe(
    model: AudioTagging,
    genres_dir: Path,
) -> pd.DataFrame:

    rows = []

    for genre, wav_path in list_audio_files(genres_dir):
        audio = load_audio_clip(wav_path)
        if audio is None:
            continue

        embedding = embed_waveform(model, audio)
        row = build_embedding_row(genre, wav_path, embedding)
        rows.append(row)

    return pd.DataFrame(rows)


def run_embedding_extraction() -> None:
    ensure_directory(EMBEDDINGS_CSV.parent)

    model = load_panns_model()
    df = generate_embeddings_dataframe(model, GENRES_DIR)

    df.to_csv(EMBEDDINGS_CSV, index=False)
    print(f"Saved embeddings to {EMBEDDINGS_CSV}, shape={df.shape}")
