"""Kurulu Ultralytics paketinin sokak örneğini entegrasyon denemesi için kopyalar."""

import argparse
from importlib.util import find_spec
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO entegrasyon örneğini hazırla")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/images/yolo_bus.jpg")
    args = parser.parse_args()
    package = find_spec("ultralytics")
    if package is None or package.origin is None:
        parser.exit(1, "Önce pip install -r requirements-yolo.txt ile YOLO bağımlılıklarını kurun.\n")
    source = Path(package.origin).parent / "assets/bus.jpg"
    if not source.is_file():
        parser.exit(1, "Kurulu Ultralytics paketinde bus.jpg örneği bulunamadı.\n")
    if args.output.exists():
        print(f"Mevcut dosya korundu: {args.output.resolve()}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, args.output)
    print(f"Sokak örneği hazır (drone verisi değildir): {args.output.resolve()}")


if __name__ == "__main__":
    main()
