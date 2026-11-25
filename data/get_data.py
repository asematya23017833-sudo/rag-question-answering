"""Download the three public arXiv papers used as the retrieval corpus.
Fetched directly from arxiv.org (authors' own public preprints).
"""
import urllib.request
from pathlib import Path

PAPERS = ["1603.02754", "1706.03762", "1810.04805"]

here = Path(__file__).resolve().parent / "pdfs"
here.mkdir(exist_ok=True)
for pid in PAPERS:
    dest = here / pid
    if dest.exists():
        print(f"{pid} already present"); continue
    print(f"downloading arXiv:{pid} ...")
    urllib.request.urlretrieve(f"https://arxiv.org/pdf/{pid}", dest)
    print(f"{pid} ok ({dest.stat().st_size:,} bytes)")
