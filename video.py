"""Kayıtlı videoyu ortak kare akışıyla işler; model bir kez yüklenir, sonuçlar akışta yazılır."""

from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import cv2

from landing_score import ScoringConfig
from obstacle_detection import YOLOObstacleDetector
from pipeline import FrameResult, process_frame
from surface_classification import SurfacePredictor
from visualization import draw_obstacles, save_image


def render_frame(frame: FrameResult, frame_index: int):
    image = draw_obstacles(cv2.cvtColor(frame.preprocessing.original_bgr, cv2.COLOR_BGR2RGB), frame.obstacles)
    best = frame.scoring.best if frame.scoring is not None else None
    if best is not None:
        for box, color in ((best.margin_box, (240, 170, 20)), (best.footprint_box, (20, 190, 240))):
            x, y, width, height = box
            cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), color, 2)
        cv2.drawMarker(image, best.center, (255, 255, 255), cv2.MARKER_CROSS, 15, 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    status = frame.scoring.status if frame.scoring else "not_scored"
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (25, 25, 25), -1)
    text = f"Frame {frame_index} | {status}" + (f" | score {best.score:.3f}" if best else "")
    cv2.putText(image, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image


def process_video(input_path, output_dir, *, options=None, stride=1, max_frames=None):
    """Her seçilen karede öneriyi yeniden hesaplar; eski kare önerisi taşınmaz."""
    if type(stride) is not int or stride < 1 or (max_frames is not None and (type(max_frames) is not int or max_frames < 1)):
        raise ValueError("Kare adımı ve azami işlenen kare sayısı pozitif tam sayı olmalı.")
    input_path, output_dir = Path(input_path), Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Video bulunamadı: {input_path}")
    final = output_dir / "annotated.avi"
    if input_path.resolve() == final.resolve():
        raise ValueError("Girdi video çıktı dosyasıyla aynı olamaz.")
    options = dict(options or {})
    if options.get("scoring_config") is None:
        options["scoring_config"] = ScoringConfig()
    detector = options.get("obstacle_detector")
    if options.get("obstacle_config") is not None and detector is None:
        options["obstacle_detector"] = YOLOObstacleDetector(options["obstacle_config"])
    if options.get("surface_config") is not None and options.get("surface_predictor") is None:
        options["surface_predictor"] = SurfacePredictor(options["surface_config"])
    capture = cv2.VideoCapture(str(input_path.resolve()))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Video açılamadı veya desteklenmeyen biçimde.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    expected_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if not isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video kare hızı okunamadı.")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "annotated.partial.avi"
    writer = None
    index = processed = selected = 0
    started = perf_counter()
    processing_seconds = 0.0
    shape = None
    try:
        with (output_dir / "frames.jsonl").open("w", encoding="utf-8") as log:
            while True:
                success, image = capture.read()
                if not success:
                    if isfinite(expected_frames) and expected_frames > 0 and index < int(expected_frames):
                        raise OSError("Video beklenen son kareden önce kesildi veya bozuk kare içeriyor.")
                    break
                frame_index = index
                index += 1
                if frame_index % stride:
                    continue
                tick = perf_counter()
                frame = process_frame(image, **options)
                elapsed = perf_counter() - tick
                processing_seconds += elapsed
                rendered = render_frame(frame, frame_index)
                if writer is None:
                    shape = image.shape[:2]
                    writer = cv2.VideoWriter(str(temporary.resolve()), cv2.VideoWriter_fourcc(*"MJPG"),
                                              fps / stride, (shape[1], shape[0]))
                    if not writer.isOpened():
                        raise OSError("Video yazıcısı açılamadı (MJPG/AVI).")
                    save_image(output_dir / "first_frame.png", rendered)
                    (output_dir / "settings.json").write_text(json.dumps(frame.settings, indent=2), encoding="utf-8")
                if image.shape[:2] != shape:
                    raise ValueError("Videoda kare boyutu değişti.")
                writer.write(rendered)
                best = frame.scoring.best
                selected += best is not None
                processed += 1
                log.write(json.dumps({"frame_index": frame_index, "timestamp_seconds": frame_index / fps,
                                      "status": frame.scoring.status, "processing_seconds": elapsed,
                                      "best": asdict(best) if best else None,
                                      "detection_count": len(frame.obstacles.detections) if frame.obstacles else None}) + "\n")
                if max_frames is not None and processed >= max_frames:
                    break
        if not processed:
            raise ValueError("Videodan işlenebilir kare okunamadı.")
        writer.release()
        writer = None
        # Başarıdan önce eksik AVI dosyası son çıktı adıyla sunulmaz.
        check = cv2.VideoCapture(str(temporary.resolve()))
        try:
            if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != processed:
                raise OSError("Kaydedilen video kare sayısı doğrulanamadı.")
        finally:
            check.release()
        temporary.replace(final)
        summary = {"status": "completed", "input": str(input_path.resolve()), "output": str(final.resolve()),
                   "processed_frames": processed, "selected_frames": selected, "stride": stride,
                   "source_fps": fps, "output_fps": fps / stride,
                   "processing_seconds": processing_seconds, "wall_seconds": perf_counter() - started,
                   "processing_fps": processed / processing_seconds, "tracking_applied": False}
    except Exception as error:
        (output_dir / "video_summary.json").write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2), encoding="utf-8")
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        temporary.unlink(missing_ok=True)
    (output_dir / "video_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
