# RoverSafe

[Türkçe](#türkçe) | [English](#english)

## Türkçe

RoverSafe, drone ve İHA görüntülerinden rover için uygun başlangıç alanları belirleyen Python tabanlı bir arazi analiz uygulamasıdır. Görüntü özelliklerini, kullanılabilir alanı ve engelleri değerlendirerek aday bölgeleri sıralar ve sonuçları görselleştirir.

### Özellikler

- Görüntü ve kayıtlı video işleme
- Gürültü azaltma, kontrast iyileştirme ve kenar/doku analizi
- Rover ayak izi ve çevre payına göre aday alan belirleme
- YOLO ile isteğe bağlı nesne tespiti
- Öğrenilmiş yüzey modeliyle zemin ayrımı
- Yükseklik haritasıyla eğim, pürüzlülük ve yerel yükseklik aralığı analizi
- İzinli bölgede alan seçimi veya sabit başlangıç noktasını değerlendirme
- Çıkış noktasına rota oluşturma
- Yapılandırılabilir parametreler ve yöntem karşılaştırmaları

RGB görüntüler görsel ön eleme için kullanılır. Metre tabanlı değerlendirme, görüntüyle aynı ızgarada yükseklik verisi gerektirir. Rover boyutları ve hareket sınırları ayarlardan belirlenir.

### Kurulum

Python 3.12 ile:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Hızlı başlangıç

Depodaki örnek görüntüyü analiz etmek için:

```powershell
.\.venv\Scripts\python.exe main.py data/samples/synthetic_terrain.png --score --output-dir outputs/demo
```

Sonuçlar `outputs/demo/` klasörüne kaydedilir.

Model veya veri indirmeden metre tabanlı sentetik örnek çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-geospatial.txt
.\.venv\Scripts\python.exe scripts/create_start_area_sample.py --output-dir outputs/sample_inputs
.\.venv\Scripts\python.exe main.py start-area --manifest outputs/sample_inputs/manifest.json --output-dir outputs/sample_result
.\.venv\Scripts\python.exe scripts/verify_start_area.py --output-dir outputs/acceptance
```

Son komut düz zemin, engel, dar geçit, eğim, eksik/belirsiz yükseklik ve sabit başlangıç dahil 12 senaryoyu denetler; JSON, CSV ve Markdown özeti üretir. Her çalıştırmada yeni veya boş çıktı klasörü kullanın. Bu sentetik kontroller yazılım davranışını sınar; gerçek arazi algılamasını veya fiziksel emniyeti doğrulamaz.

### Proje yapısı

| Klasör | İçerik |
| --- | --- |
| `configs/` | İşlem ve deney ayarları |
| `data/` | Örnek görüntü, veri kaynakları ve manifestler |
| `models/` | Model açıklamaları ve eğitim kayıtları |
| `scripts/` | Veri hazırlama, eğitim ve deney araçları |
| `tests/` | Otomatik testler |
| `outputs/` | Çalıştırma çıktıları |

### Dokümantasyon

- [Kullanım ve çalışma seçenekleri](docs/KULLANIM.md)
- [Veri kaynakları](data/SOURCES.md)
- [Modeller](models/README.md)

Büyük veri dosyaları, model ağırlıkları ve üretilen çıktılar depoya eklenmez.

---

## English

RoverSafe is a Python-based terrain analysis application that identifies suitable starting areas for rovers from drone and UAV imagery. It evaluates image features, available space, and obstacles to rank candidate regions and visualize the results.

### Features

- Image and recorded video processing
- Noise reduction, contrast enhancement, and edge/texture analysis
- Candidate region extraction based on rover footprint and clearance margin
- Optional object detection with YOLO
- Ground classification using a trained surface model
- Slope, roughness, and local height range analysis from elevation maps
- Site selection within an allowed region or assessment of a fixed starting point
- Route generation to an exit point
- Configurable parameters and method comparisons

RGB imagery is used for visual screening. Metric assessment requires elevation data aligned to the same grid as the image. Rover dimensions and mobility limits are configurable.

### Installation

Using Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Quick start

Analyze the sample image included in the repository:

```powershell
.\.venv\Scripts\python.exe main.py data/samples/synthetic_terrain.png --score --output-dir outputs/demo
```

Results are saved to `outputs/demo/`.

Run a metric synthetic example without downloading models or datasets:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-geospatial.txt
.\.venv\Scripts\python.exe scripts/create_start_area_sample.py --output-dir outputs/sample_inputs
.\.venv\Scripts\python.exe main.py start-area --manifest outputs/sample_inputs/manifest.json --output-dir outputs/sample_result
.\.venv\Scripts\python.exe scripts/verify_start_area.py --output-dir outputs/acceptance
```

The final command checks 12 scenarios covering flat terrain, obstacles, narrow passages, slopes, missing/uncertain elevation, and fixed starts; it writes JSON, CSV, and Markdown summaries. Use a new or empty output directory for each run. These synthetic checks verify software behavior, not real terrain perception or physical safety.

### Project structure

| Directory | Contents |
| --- | --- |
| `configs/` | Processing and experiment settings |
| `data/` | Sample image, data sources, and manifests |
| `models/` | Model descriptions and training metadata |
| `scripts/` | Data preparation, training, and experiment tools |
| `tests/` | Automated tests |
| `outputs/` | Generated outputs |

### Documentation

- [Usage and options (Turkish)](docs/KULLANIM.md)
- [Data sources (Turkish)](data/SOURCES.md)
- [Models (Turkish)](models/README.md)

Large datasets, model weights, and generated outputs are not included in the repository.
