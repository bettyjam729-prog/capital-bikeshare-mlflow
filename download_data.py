"""Download and extract the official UCI daily bike-sharing dataset."""
from pathlib import Path
import io
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

def main():
    target = Path("data/day.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        with zf.open("day.csv") as src:
            target.write_bytes(src.read())
    print(f"Saved {target} ({target.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
