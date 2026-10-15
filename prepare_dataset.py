"""Recreate the project label subset from extracted FMA files.

Expected folders in the repository root:
  fma_small/
  fma_metadata/tracks.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
TRACKS_CSV = ROOT / "fma_metadata" / "tracks.csv"
AUDIO_ROOT = ROOT / "fma_small"
OUTPUT = ROOT / "data" / "song_labels.csv"

if not TRACKS_CSV.exists():
    raise FileNotFoundError(f"Could not find {TRACKS_CSV}. Download and extract FMA metadata first.")
if not AUDIO_ROOT.exists():
    raise FileNotFoundError(f"Could not find {AUDIO_ROOT}. Download and extract FMA Small audio first.")

tracks = pd.read_csv(TRACKS_CSV, header=[0, 1], index_col=0)
df = pd.DataFrame({
    "track_id": tracks.index,
    "title": tracks["track"]["title"],
    "listens": tracks["track"]["listens"],
    "genre": tracks["track"]["genre_top"],
})
df["listens"] = pd.to_numeric(df["listens"], errors="coerce")
df = df.dropna(subset=["listens", "title"])
df = df[df["listens"] > 0].copy()

def audio_path(track_id):
    track_id = int(track_id)
    return AUDIO_ROOT / f"{track_id // 1000:03d}" / f"{track_id:06d}.mp3"

df["audio_path"] = df["track_id"].apply(audio_path)
df = df[df["audio_path"].apply(Path.exists)].copy()
df = df.sort_values("listens", ascending=False).reset_index(drop=True)

n = 500
if len(df) < 2 * n:
    raise ValueError(f"Only {len(df)} tracks with audio and positive listen counts found; 1,000 are required.")
high = df.head(n).copy()
low = df.tail(n).copy()
high["label"] = 1
low["label"] = 0
subset = pd.concat([high, low], ignore_index=True)
subset = subset.sample(frac=1, random_state=42).reset_index(drop=True)
subset["audio_path"] = subset["audio_path"].apply(lambda p: str(Path(p).relative_to(ROOT)))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
subset[["track_id", "title", "listens", "genre", "audio_path", "label"]].to_csv(OUTPUT, index=False)
print(f"Saved {len(subset)} rows to {OUTPUT}")
print(subset["label"].value_counts().sort_index().rename(index={0: "less popular", 1: "popular"}))
