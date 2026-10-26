"""Obtain checksum-verified, unmodified Doom shareware 1.9 assets."""

import argparse
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doomnet.wad import SHAREWARE_MD5, SHAREWARE_URL, verify_shareware


def fetch_shareware(destination: str | Path, url: str = SHAREWARE_URL) -> Path:
    path = Path(destination)
    if path.exists():
        verify_shareware(path.read_bytes())
        return path
    request = urllib.request.Request(url, headers={"User-Agent": "doomnet-shareware-fetch/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    verify_shareware(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/doom1.wad"))
    parser.add_argument("--url", default=SHAREWARE_URL,
                        help="Original shareware mirror; the exact checksum is always required")
    args = parser.parse_args()
    path = fetch_shareware(args.output, args.url)
    print(f"Verified {path}: Doom shareware 1.9, MD5 {SHAREWARE_MD5}")


if __name__ == "__main__":
    main()
