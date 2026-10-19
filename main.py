"""RGB ön elemesini veya metre tabanlı rover başlangıç analizini çalıştırır."""

import argparse
import json
from pathlib import Path

import cv2

from pipeline import process_image_file
from landing_score import ScoringConfig
from obstacle_detection import ObstacleConfig, ObstacleDetectionError
from segmentation import SegmentationConfig
from terrain_analysis import TerrainConfig
from configuration import cli_defaults
from surface_classification import SurfaceConfig


def main() -> int:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'start-area':
        from start_area_cli import main as start_area_main
        return start_area_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description="RoverSafe görsel aday alan analizi; metre tabanlı mod için start-area kullanın")
    parser.add_argument("image", type=Path, help="İşlenecek görüntünün yolu")
    parser.add_argument("--config", type=Path, help="JSON işlem ayarları; açık CLI seçenekleri önceliklidir")
    parser.add_argument("--video", action="store_true", help="Girdiyi kayıtlı video olarak işle")
    parser.add_argument("--stride", type=int, default=1, help="Videoda her N'inci kareyi işle")
    parser.add_argument("--max-frames", type=int, help="Videoda en fazla işlenecek kare sayısı")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/preprocessing"))
    parser.add_argument("--filter", choices=["gaussian", "median", "none"], default="gaussian")
    parser.add_argument("--kernel-size", type=int, default=5, help="3 veya daha büyük tek sayı")
    parser.add_argument("--clahe", action="store_true", help="Yerel kontrast iyileştirmeyi aç")
    terrain_group = parser.add_argument_group("Arazi analizi (--terrain, --candidates veya --score)")
    terrain_group.add_argument("--terrain", action="store_true", help="Kenar ve doku haritalarını çıkar")
    terrain_group.add_argument("--window-size", type=int, help="Yerel analiz penceresi; varsayılan 25")
    terrain_group.add_argument("--edge-method", choices=["canny", "sobel"], help="Yoğunluk yöntemi; varsayılan canny")
    terrain_group.add_argument("--canny-low", type=int, help="Canny alt eşiği; varsayılan 50")
    terrain_group.add_argument("--canny-high", type=int, help="Canny üst eşiği; varsayılan 150")
    terrain_group.add_argument("--sobel-threshold", type=float, help="Ham Sobel büyüklüğü eşiği; varsayılan 80")
    candidate_group = parser.add_argument_group("Aday bölgeler (--candidates veya --score)")
    candidate_group.add_argument("--candidates", action="store_true", help="Arazi analizi ve aday bölge çıkarmayı aç")
    candidate_group.add_argument("--max-edge-density", type=float, help="En yüksek yerel yoğunluk; varsayılan 0.03")
    candidate_group.add_argument("--max-variance", type=float, help="En yüksek yerel varyans; varsayılan 250")
    candidate_group.add_argument("--min-area", type=int, help="En küçük bileşen alanı (px²); varsayılan 2500")
    candidate_group.add_argument("--opening-size", type=int, help="Morfolojik açma; varsayılan 3, kapatmak için 1")
    candidate_group.add_argument("--rover-width", type=int, help="Ayak izi genişliği (px); varsayılan 80")
    candidate_group.add_argument("--rover-height", type=int, help="Ayak izi yüksekliği (px); varsayılan 60")
    candidate_group.add_argument("--safety-margin", type=int, help="Her yönde pay (px); varsayılan 10")
    scoring_group = parser.add_argument_group("Puanlama (--score ile açılır)")
    scoring_group.add_argument("--score", action="store_true", help="Adayları puanla ve öneri seç")
    scoring_group.add_argument("--edge-weight", type=float, help="Kenar seyrekliği ağırlığı; varsayılan 0.3")
    scoring_group.add_argument("--texture-weight", type=float, help="Doku homojenliği ağırlığı; varsayılan 0.3")
    scoring_group.add_argument("--clearance-weight", type=float, help="Çevredeki boşluk ağırlığı; varsayılan 0.3")
    scoring_group.add_argument("--area-weight", type=float, help="Bölge büyüklüğü ağırlığı; varsayılan 0.1")
    scoring_group.add_argument("--clearance-target", type=float, help="Ek boşluk hedefi (px); varsayılan 40")
    scoring_group.add_argument("--area-target-multiple", type=float, help="Alan hedefi / paylı ayak izi; varsayılan 4")
    scoring_group.add_argument("--min-score", type=float, help="Öneri için asgari puan; varsayılan 0.5")
    obstacle_group = parser.add_argument_group("YOLO engel eleme (--obstacles ile tüm akış açılır)")
    obstacle_group.add_argument("--obstacles", action="store_true", help="Nesne tespiti, aday eleme ve öneriyi aç")
    obstacle_group.add_argument("--yolo-model", help="Yerel model yolu; varsayılan models/yolo11n.pt")
    obstacle_group.add_argument("--yolo-confidence", type=float, help="En düşük tespit güveni; varsayılan 0.25")
    obstacle_group.add_argument("--yolo-iou", type=float, help="NMS IoU eşiği; varsayılan 0.7")
    obstacle_group.add_argument("--obstacle-margin", type=int, help="Kutu çevresindeki ek pay (px); varsayılan 15")
    obstacle_group.add_argument("--yolo-image-size", type=int, help="YOLO giriş boyutu; varsayılan 640")
    obstacle_group.add_argument("--yolo-device", help="Çalışma cihazı; varsayılan cpu")
    obstacle_group.add_argument("--obstacle-classes", nargs="+", help="İngilizce sınıf adları; verilmezse tüm sınıflar")
    parser.add_argument("--surface", action="store_true", help="Öğrenilmiş renk/doku yüzey filtresini aç")
    parser.add_argument("--surface-model", help="Yerel yüzey modeli yolu")
    parser.add_argument("--surface-threshold", type=float, help="Yüzey sınıf puanı eşiği (güvenlik olasılığı değildir)")
    parser.add_argument("--surface-grid-step", type=int, help="Yüzey özellik örnekleme adımı; varsayılan 4")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config is not None:
        try:
            parser.set_defaults(**cli_defaults(preliminary.config))
        except (OSError, ValueError) as error:
            parser.error(str(error))
    args = parser.parse_args()
    terrain_options = {
        name: getattr(args, name)
        for name in ("window_size", "edge_method", "canny_low", "canny_high", "sobel_threshold")
        if getattr(args, name) is not None
    }
    candidate_options = {
        name: getattr(args, name)
        for name in ("max_edge_density", "max_variance", "min_area", "opening_size",
                     "rover_width", "rover_height", "safety_margin")
        if getattr(args, name) is not None
    }
    scoring_options = {
        name: getattr(args, name)
        for name in ("edge_weight", "texture_weight", "clearance_weight", "area_weight",
                     "clearance_target", "area_target_multiple", "min_score")
        if getattr(args, name) is not None
    }
    obstacle_options = {field: getattr(args, option) for field, option in (
        ("model_path", "yolo_model"), ("confidence", "yolo_confidence"), ("iou", "yolo_iou"),
        ("margin", "obstacle_margin"), ("image_size", "yolo_image_size"), ("device", "yolo_device"),
    ) if getattr(args, option) is not None}
    if args.obstacle_classes is not None:
        obstacle_options["classes"] = tuple(args.obstacle_classes)
    surface_options = {field: getattr(args, option) for field, option in (
        ("model_path", "surface_model"), ("threshold", "surface_threshold"), ("grid_step", "surface_grid_step"))
        if getattr(args, option) is not None}
    if surface_options and not args.surface:
        parser.error("Yüzey ayarları için --surface gerekli.")
    terrain_enabled = args.terrain or args.candidates or args.score or args.obstacles or args.surface
    candidates_enabled = args.candidates or args.score or args.obstacles or args.surface
    scoring_enabled = args.score or args.obstacles or args.video or args.surface
    if args.video:
        terrain_enabled = candidates_enabled = True
    elif args.stride != 1 or args.max_frames is not None:
        parser.error("Kare adımı ve sınırı için --video seçeneği gerekli.")
    if terrain_options and not terrain_enabled:
        parser.error("Arazi ayarları için --terrain, --candidates, --score veya --obstacles ekleyin.")
    if candidate_options and not candidates_enabled:
        parser.error("Aday ayarları için --candidates, --score veya --obstacles ekleyin.")
    if scoring_options and not scoring_enabled:
        parser.error("Puanlama ayarları için --score veya --obstacles ekleyin.")
    if obstacle_options and not args.obstacles:
        parser.error("YOLO ayarları için --obstacles seçeneğini ekleyin.")
    try:
        terrain_config = TerrainConfig(**terrain_options) if terrain_enabled else None
        segmentation_config = SegmentationConfig(**candidate_options) if candidates_enabled else None
        scoring_config = ScoringConfig(**scoring_options) if scoring_enabled else None
        obstacle_config = ObstacleConfig(**obstacle_options) if args.obstacles else None
        surface_config = SurfaceConfig(**surface_options) if args.surface else None
        if args.video:
            from video import process_video
            summary = process_video(args.image, args.output_dir, stride=args.stride, max_frames=args.max_frames,
                                    options={"filter_type": args.filter, "kernel_size": args.kernel_size,
                                             "use_clahe": args.clahe, "terrain_config": terrain_config,
                                             "segmentation_config": segmentation_config, "scoring_config": scoring_config,
                                             "obstacle_config": obstacle_config, "surface_config": surface_config})
            print(f"Video tamamlandı: {summary['processed_frames']} kare · {summary['output']}")
            return 0
        comparison = process_image_file(
            args.image, args.output_dir, args.filter, args.kernel_size, args.clahe,
            terrain_config=terrain_config,
            segmentation_config=segmentation_config,
            scoring_config=scoring_config,
            obstacle_config=obstacle_config,
            surface_config=surface_config,
        )
    except (OSError, ValueError, cv2.error, ObstacleDetectionError) as error:
        parser.exit(1, f"Hata: {error}\n")
    print(f"Ön işleme tamamlandı. Karşılaştırma: {comparison.resolve()}")
    if terrain_enabled:
        print(f"Arazi analizi: {(args.output_dir / 'terrain_comparison.png').resolve()}")
    if candidates_enabled:
        print(f"Aday bölgeler: {(args.output_dir / 'candidate_comparison.png').resolve()}")
    if args.obstacles:
        detections = json.loads((args.output_dir / "obstacle_detections.json").read_text(encoding="utf-8"))
        print(f"YOLO: {detections['detection_count']} nesne · dışlanan piksel: {detections['excluded_pixel_count']}")
    if scoring_enabled:
        summary = json.loads((args.output_dir / "landing_recommendation.json").read_text(encoding="utf-8"))
        if summary["best"] is not None:
            print(f"Önerilen merkez: {summary['best']['center']} · puan: {summary['best']['score']:.3f}")
        else:
            print("Öneri yok: " + ("asgari puan karşılanmadı." if summary["status"] == "below_min_score"
                                  else "geçerli aday bulunamadı."))
        print(f"Öneri görseli: {(args.output_dir / 'landing_comparison.png').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
