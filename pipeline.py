"""Dosya okuma, kare ön işleme ve sonuç kaydetme adımlarını birleştirir."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from preprocessing import PreprocessingResult, preprocess_image, read_image
from landing_score import ScoringConfig, ScoringResult, score_landing
from obstacle_detection import ObstacleConfig, ObstacleResult, YOLOObstacleDetector
from segmentation import SegmentationConfig, SegmentationResult, extract_candidates
from terrain_analysis import TerrainConfig, TerrainResult, analyze_terrain
from surface_classification import SurfaceConfig, SurfacePredictor, SurfaceResult
from visualization import save_candidate_results, save_obstacle_results, save_results, save_scoring_results, save_terrain_results


@dataclass
class FrameResult:
    preprocessing: PreprocessingResult
    terrain: TerrainResult | None
    candidates: SegmentationResult | None
    scoring: ScoringResult | None
    obstacles: ObstacleResult | None
    settings: dict
    surface: SurfaceResult | None = None


def process_frame(
    image_bgr,
    filter_type: str = "gaussian",
    kernel_size: int = 5,
    use_clahe: bool = False,
    *,
    terrain_config: TerrainConfig | None = None,
    segmentation_config: SegmentationConfig | None = None,
    scoring_config: ScoringConfig | None = None,
    obstacle_config: ObstacleConfig | None = None,
    obstacle_detector: YOLOObstacleDetector | None = None,
    surface_config: SurfaceConfig | None = None,
    surface_predictor: SurfacePredictor | None = None,
) -> FrameResult:
    """Dosya veya görsel yazmadan bir kareyi işler; video ve deneylerde ortak akış."""
    if obstacle_detector is not None:
        if obstacle_config is not None and obstacle_config != obstacle_detector.config:
            raise ValueError("Verilen algılayıcı ile engel ayarları aynı olmalı.")
        obstacle_config = obstacle_detector.config
    if surface_predictor is not None:
        if surface_config is not None and surface_config != surface_predictor.config:
            raise ValueError("Yüzey modeli ile ayarları aynı olmalı.")
        surface_config = surface_predictor.config
    if (obstacle_config is not None or surface_config is not None) and scoring_config is None:
        scoring_config = ScoringConfig()
    if scoring_config is not None and segmentation_config is None:
        segmentation_config = SegmentationConfig()
    if segmentation_config is not None and terrain_config is None:
        terrain_config = TerrainConfig()
    result = preprocess_image(image_bgr, filter_type, kernel_size, use_clahe)
    obstacles = None
    if obstacle_config is not None:
        detector = obstacle_detector if obstacle_detector is not None else YOLOObstacleDetector(obstacle_config)
        obstacles = detector.detect(result.original_bgr)
    surface = None
    if surface_config is not None:
        predictor = surface_predictor if surface_predictor is not None else SurfacePredictor(surface_config)
        surface = predictor.predict(result.original_bgr)
    exclusion = obstacles.exclusion_mask.copy() if obstacles is not None else None
    if surface is not None:
        exclusion = surface.exclusion_mask.copy() if exclusion is None else exclusion | surface.exclusion_mask
    # Hatalı analiz ayarlarında yarım çıktı üretmeden önce doğrulama yapılır.
    terrain = analyze_terrain(result.processed, terrain_config) if terrain_config is not None else None
    candidates = extract_candidates(
        terrain, segmentation_config,
        exclusion_mask=exclusion,
    ) if segmentation_config is not None else None
    scoring = score_landing(terrain, candidates, scoring_config) if scoring_config is not None else None
    settings = {
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "filter_type": filter_type,
        "kernel_size": kernel_size,
        "use_clahe": use_clahe,
        "terrain": asdict(terrain_config) if terrain_config is not None else None,
        "segmentation": asdict(segmentation_config) if segmentation_config is not None else None,
        "scoring": asdict(scoring_config) if scoring_config is not None else None,
        "obstacles": asdict(obstacle_config) if obstacle_config is not None else None,
        "surface": asdict(surface_config) if surface_config is not None else None,
        "surface_model_sha256": surface.model_sha256 if surface is not None else None,
    }
    return FrameResult(result, terrain, candidates, scoring, obstacles, settings, surface)


def process_image_file(
    input_path: str | Path, output_dir: str | Path,
    filter_type: str = "gaussian", kernel_size: int = 5, use_clahe: bool = False,
    *, terrain_config: TerrainConfig | None = None,
    segmentation_config: SegmentationConfig | None = None,
    scoring_config: ScoringConfig | None = None,
    obstacle_config: ObstacleConfig | None = None,
    obstacle_detector: YOLOObstacleDetector | None = None,
    surface_config: SurfaceConfig | None = None,
    surface_predictor: SurfacePredictor | None = None,
) -> Path:
    """Tek görüntüyü ortak kare akışından geçirir ve ara çıktıları kaydeder."""
    frame = process_frame(read_image(input_path), filter_type, kernel_size, use_clahe,
                          terrain_config=terrain_config, segmentation_config=segmentation_config,
                          scoring_config=scoring_config, obstacle_config=obstacle_config,
                          obstacle_detector=obstacle_detector, surface_config=surface_config,
                          surface_predictor=surface_predictor)
    result, terrain, candidates, scoring, obstacles = (
        frame.preprocessing, frame.terrain, frame.candidates, frame.scoring, frame.obstacles)
    settings = dict(frame.settings, input_path=str(Path(input_path).resolve()))
    comparison = save_results(result, output_dir)
    if frame.surface is not None:
        from visualization import save_image
        import numpy as np
        save_image(Path(output_dir) / "surface_score.png", (frame.surface.probability * 255).astype(np.uint8))
        save_image(Path(output_dir) / "surface_exclusion.png", frame.surface.exclusion_mask.astype(np.uint8) * 255)
    if obstacles is not None:
        save_obstacle_results(result.original_bgr, obstacles, output_dir)
        summary = {
            "status": "completed", "detection_count": len(obstacles.detections),
            "excluded_pixel_count": int(obstacles.exclusion_mask.sum()),
            "detections": [asdict(item) for item in obstacles.detections],
            "inference_seconds": obstacles.inference_seconds,
            "model_path": obstacles.model_path, "model_sha256": obstacles.model_sha256,
            "ultralytics_version": obstacles.ultralytics_version,
        }
        (Path(output_dir) / "obstacle_detections.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    if terrain is not None:
        save_terrain_results(result.original_bgr, terrain, output_dir)
        summary = {
            "edge_method": terrain.config.edge_method,
            "edge_density_mean": float(terrain.edge_density.mean()),
            "local_variance_mean": float(terrain.local_variance.mean()),
            "local_variance_max": float(terrain.local_variance.max()),
            "sobel_edge_fraction": float((terrain.sobel_edges > 0).mean()),
            "canny_edge_fraction": float((terrain.canny_edges > 0).mean()),
        }
        (Path(output_dir) / "terrain_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if candidates is not None:
        save_candidate_results(result.original_bgr, candidates, output_dir, obstacles=obstacles)
        summary = {
            "status": "candidates_found" if candidates.regions else "no_candidate",
            "candidate_count": len(candidates.regions),
            "valid_center_count": int(candidates.valid_centers.sum()),
            "rejected_small": candidates.rejected_small,
            "rejected_no_fit": candidates.rejected_no_fit,
            "regions": [asdict(region) for region in candidates.regions],
        }
        (Path(output_dir) / "candidates.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if scoring is not None:
        save_scoring_results(result.original_bgr, scoring, output_dir, obstacles=obstacles)
        summary = {
            "status": scoring.status,
            "best": asdict(scoring.best) if scoring.best is not None else None,
            "ranked_regions": [asdict(point) for point in scoring.ranked_regions],
            "min_score": scoring.config.min_score,
            "obstacle_detection_applied": obstacles is not None,
            "surface_filter_applied": frame.surface is not None,
        }
        (Path(output_dir) / "landing_recommendation.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    (Path(output_dir) / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return comparison
