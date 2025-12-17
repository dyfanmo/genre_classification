from pathlib import Path

DATA_DIR: Path = Path("Data")

GENRES_DIR: Path = DATA_DIR / "genres_original"

EMBEDDINGS_CSV: Path = Path("data/embeddings/panns_embeddings.csv")

SAMPLE_RATE: int = 32000
CLIP_DURATION_SECONDS: float = 10.0
