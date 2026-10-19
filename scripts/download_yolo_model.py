"""Resmî Ultralytics sürümünden küçük YOLO11n modelini yerel models klasörüne indirir."""

import argparse
from pathlib import Path
import urllib.request


MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="RoverSafe YOLO11n modelini indir")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "models/yolo11n.pt")
    args = parser.parse_args()
    if args.output.exists():
        print(f"Mevcut dosya korundu: {args.output.resolve()}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.part")
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "RoverSafe/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if temporary.stat().st_size < 1024:
            raise ValueError("İndirilen dosya beklenenden küçük.")
        temporary.replace(args.output)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        parser.exit(1, f"Model indirilemedi: {error}\n")
    print(f"Model hazır: {args.output.resolve()}")


if __name__ == "__main__":
    main()
