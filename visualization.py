"""Ön işleme, arazi özellikleri ve aday bölgeleri görsel/sayısal olarak kaydeder."""

from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import PowerNorm
from matplotlib.figure import Figure

from preprocessing import PreprocessingResult
from landing_score import ScoringResult
from obstacle_detection import ObstacleResult
from segmentation import SegmentationResult
from terrain_analysis import SOBEL_DISPLAY_MAX, VARIANCE_MAX, TerrainResult


def save_image(path: str | Path, image: np.ndarray) -> None:
    """PNG/JPEG verisini Unicode dosya yollarında güvenilir biçimde yazar."""
    path = Path(path)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"Görüntü kodlanamadı: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def save_results(result: PreprocessingResult, output_dir: str | Path) -> Path:
    """Dört ara görüntüyü ve tek bir karşılaştırma figürünü kaydeder."""
    output_dir = Path(output_dir)
    images = {
        "original.png": result.original_bgr,
        "grayscale.png": result.grayscale,
        "denoised.png": result.denoised,
        "processed.png": result.processed,
    }
    for name, image in images.items():
        save_image(output_dir / name, image)

    # Agg tuvali pencere açmadan, masaüstü ve sunucu ortamlarında çalışır.
    figure = Figure(figsize=(12, 7), layout="constrained")
    FigureCanvasAgg(figure)
    panels = [
        (cv2.cvtColor(result.original_bgr, cv2.COLOR_BGR2RGB), "Orijinal görüntü"),
        (result.grayscale, "Gri seviye"),
        (result.denoised, "Gürültü azaltma"),
        (result.processed, "Son ön işleme çıktısı"),
    ]
    for axis, (image, title) in zip(figure.subplots(2, 2).flat, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    comparison = output_dir / "comparison.png"
    figure.savefig(comparison, dpi=140)
    return comparison


def save_candidate_results(
    original_bgr: np.ndarray, result: SegmentationResult, output_dir: str | Path,
    *, obstacles: ObstacleResult | None = None,
) -> Path:
    """Adayları, izin verilen merkezleri ve puanlanmamış bölge etiketlerini saklar."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in (
        ("threshold_mask", result.threshold_mask), ("cleaned_mask", result.cleaned_mask),
        ("candidate_mask", result.candidate_mask), ("valid_centers", result.valid_centers),
    ):
        save_image(output_dir / f"{name}.png", mask.astype(np.uint8) * 255)
    np.savez_compressed(
        output_dir / "candidate_regions.npz",
        labels=result.labels, valid_centers=result.valid_centers,
        threshold_mask=result.threshold_mask, cleaned_mask=result.cleaned_mask,
        candidate_mask=result.candidate_mask,
    )
    overlay = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    # Sarı aday, turkuaz geçerli merkezdir; renkler güvenlik sınıfları değildir.
    for mask, color in ((result.candidate_mask, (245, 183, 40)), (result.valid_centers, (25, 200, 220))):
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array(color)).astype(np.uint8)
    overlay = draw_obstacles(overlay, obstacles)
    save_image(output_dir / "candidate_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    figure = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(figure)
    panels = [
        (overlay, f"{len(result.regions)} aday bölge · sarı: bölge, turkuaz: merkez"),
        (result.threshold_mask, "Kenar ve varyans eşiklerinden geçen pikseller"),
        (result.cleaned_mask, "Morfolojik açma sonrası"),
        (result.valid_centers, "Rover + güvenlik payının sığdığı merkezler"),
    ]
    axes = figure.subplots(2, 2)
    for axis, (data, title) in zip(axes.flat, panels):
        axis.imshow(data, cmap="gray", vmin=0, vmax=1 if data.ndim == 2 else 255)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    for region in result.regions:
        x, y, _, _ = region.bounding_box
        axes[0, 0].text(x + 3, y + 12, f"A{region.region_id}", color="black", fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1})
    config = result.config
    status = "Aday bulunamadı" if not result.regions else "Adaylar henüz puanlanmadı"
    figure.suptitle(
        f"{status} · rover {config.rover_width} × {config.rover_height} px · pay {config.safety_margin} px\n"
        + (f"YOLO: {len(obstacles.detections)} nesne · kırmızı: dışlanan engel/pay"
         if obstacles is not None else "Görsel ve geometrik adaylar — nesne tespiti uygulanmadı"),
        fontsize=12,
    )
    comparison = output_dir / "candidate_comparison.png"
    figure.savefig(comparison, dpi=140)
    return comparison


def save_scoring_results(
    original_bgr: np.ndarray, result: ScoringResult, output_dir: str | Path,
    *, obstacles: ObstacleResult | None = None,
) -> Path:
    """Seçilen ayak izini ve yalnızca geçerli merkezlerdeki puan haritasını gösterir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "landing_scores.npz", score_map=result.score_map,
                        clearance_map=result.clearance_map)
    overlay = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    overlay = draw_obstacles(overlay, obstacles)
    if result.best is not None:
        for box, color in ((result.best.margin_box, (240, 170, 20)),
                           (result.best.footprint_box, (20, 190, 240))):
            x, y, width, height = box
            cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), color, 2)
        cv2.drawMarker(overlay, result.best.center, (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
    save_image(output_dir / "landing_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    figure = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    axes[0].imshow(overlay)
    axes[0].set_title("Mavi: rover · sarı: pay dahil ayak izi")
    cmap = colormaps["viridis"].copy()
    cmap.set_bad("#dedede")
    heatmap = axes[1].imshow(np.ma.masked_invalid(result.score_map), cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Geçerli merkez puanları · gri: geçersiz")
    figure.colorbar(heatmap, ax=axes[1], label="Uygunluk puanı (0–1)", shrink=0.8)
    for axis in axes:
        axis.axis("off")
    if result.best is not None:
        title = f"Öneri: {result.best.center} · A{result.best.region_id} · puan {result.best.score:.3f}"
    elif result.status == "below_min_score":
        title = "Öneri yok: aday puanları asgari eşiğin altında"
    else:
        title = "Öneri yok: geçerli aday bulunamadı"
    detection_status = (f"YOLO: {len(obstacles.detections)} nesne · kırmızı: engel/pay"
                        if obstacles is not None else "Nesne tespiti uygulanmadı")
    figure.suptitle(f"{title}\nGörsel uygunluk önerisi · {detection_status}", fontsize=12)
    comparison = output_dir / "landing_comparison.png"
    figure.savefig(comparison, dpi=140)
    return comparison


def draw_obstacles(image_rgb: np.ndarray, obstacles: ObstacleResult | None) -> np.ndarray:
    """Tespit kutularını ve genişletilmiş dışlama alanlarını RGB görüntüye çizer."""
    if obstacles is None:
        return image_rgb
    image_rgb = image_rgb.copy()
    mask = obstacles.exclusion_mask
    image_rgb[mask] = (0.6 * image_rgb[mask] + 0.4 * np.array((240, 35, 35))).astype(np.uint8)
    for obstacle in obstacles.detections:
        x, y, width, height = obstacle.bounding_box
        cv2.rectangle(image_rgb, (x, y), (x + width - 1, y + height - 1), (255, 45, 45), 2)
        cv2.putText(image_rgb, f"{obstacle.class_name} {obstacle.confidence:.2f}",
                    (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 45, 45), 1)
    return image_rgb


def save_obstacle_results(original_bgr: np.ndarray, result: ObstacleResult, output_dir: str | Path) -> None:
    """Paylı dışlama maskesini ve özgün boyuttaki nesne tespit görselini kaydeder."""
    output_dir = Path(output_dir)
    save_image(output_dir / "obstacle_mask.png", result.exclusion_mask.astype(np.uint8) * 255)
    overlay = draw_obstacles(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), result)
    save_image(output_dir / "obstacle_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    np.savez_compressed(output_dir / "obstacles.npz", exclusion_mask=result.exclusion_mask)


def save_terrain_results(
    original_bgr: np.ndarray, result: TerrainResult, output_dir: str | Path
) -> Path:
    """Kenarları, sabit ölçekli haritaları ve kayıpsız sayısal dizileri kaydeder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(output_dir / "sobel_edges.png", result.sobel_edges)
    save_image(output_dir / "canny_edges.png", result.canny_edges)
    # PNG ön izleme içindir; sonraki hesaplamalar yuvarlanmamış NPZ'yi kullanır.
    save_image(
        output_dir / "edge_density.png", np.rint(result.edge_density * 255).astype(np.uint8)
    )
    save_image(
        output_dir / "local_variance.png",
        np.rint(result.local_variance / VARIANCE_MAX * 255).astype(np.uint8),
    )
    np.savez_compressed(
        output_dir / "terrain_features.npz",
        sobel_magnitude=result.sobel_magnitude,
        sobel_edges=result.sobel_edges,
        canny_edges=result.canny_edges,
        edge_density=result.edge_density,
        local_variance=result.local_variance,
    )

    figure = Figure(figsize=(14, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3).flat
    panels = [
        (cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), "Orijinal görüntü", None, None),
        (result.sobel_magnitude, "Sobel gradyan büyüklüğü", "magma", SOBEL_DISPLAY_MAX),
        (result.sobel_edges, "Sobel kenarları", "gray", 255),
        (result.canny_edges, "Canny kenarları", "gray", 255),
        (result.edge_density, f"Kenar yoğunluğu ({result.config.edge_method})", "magma", 1),
        (result.local_variance, "Yerel varyans (gri seviye²)", "magma", VARIANCE_MAX),
    ]
    for axis, (data, title, cmap, maximum) in zip(axes, panels):
        if cmap == "magma":
            # Sabit karekök ölçeği, küçük değerleri görünür tutarken sahneler arası
            # renk karşılaştırmasını korur. Ham sayısal değerler değişmez.
            plot = axis.imshow(data, cmap=cmap, norm=PowerNorm(0.5, vmin=0, vmax=maximum))
        else:
            plot = axis.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=maximum)
        axis.set_title(title)
        axis.axis("off")
        if cmap == "magma":
            figure.colorbar(plot, ax=axis, shrink=0.75)
    figure.suptitle(
        f"Arazi özellikleri · {result.config.window_size} × {result.config.window_size} pencere\n"
        "Renkler karekök ölçeğinde · Haritalar görsel karmaşıklığı gösterir",
        fontsize=12,
    )
    comparison = output_dir / "terrain_comparison.png"
    figure.savefig(comparison, dpi=140)
    return comparison
