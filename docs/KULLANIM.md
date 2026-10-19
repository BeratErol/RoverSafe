# Kullanım

Komutlar proje kökünden çalıştırılır. Örneklerde Windows sanal ortam yorumlayıcısı kullanılır. Linux ve macOS üzerinde karşılığı `.venv/bin/python` yoludur.

## Görüntü analizi

```powershell
.\.venv\Scripts\python.exe main.py data/images/arazi.jpg --score --output-dir outputs/image
```

Ön işleme seçenekleri:

```powershell
.\.venv\Scripts\python.exe main.py data/images/arazi.jpg --score --filter median --kernel-size 3 --clahe --output-dir outputs/filtered
```

JSON ayar dosyasıyla çalıştırma:

```powershell
.\.venv\Scripts\python.exe main.py data/images/arazi.jpg --config configs/default.json --score --output-dir outputs/configured
```

## Video analizi

```powershell
.\.venv\Scripts\python.exe main.py data/videos/arazi.mp4 --video --score --stride 2 --max-frames 100 --output-dir outputs/video
```

`--stride` kare örnekleme aralığını, `--max-frames` işlenecek en fazla kare sayısını belirler.

## Nesne tespiti

YOLO bağımlılıklarını ve ağırlığını hazırlayın:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt
.\.venv\Scripts\python.exe scripts/download_yolo_model.py
.\.venv\Scripts\python.exe main.py data/images/arazi.jpg --obstacles --output-dir outputs/obstacles
```

Algılanan nesnelerin çevresindeki paylı bölgeler adaylardan çıkarılır.

## Yüzey modeli

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-surface.txt
.\.venv\Scripts\python.exe scripts/train_surface.py
.\.venv\Scripts\python.exe main.py data/images/arazi.jpg --surface --config configs/surface.json --output-dir outputs/surface
```

Eğitimden önce veri indirme araçlarıyla görüntü ve maskeleri hazırlayın. Model seçenekleri [model açıklamasında](../models/README.md) yer alır.

## Metre tabanlı başlangıç alanı analizi

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-geospatial.txt
.\.venv\Scripts\python.exe main.py start-area --manifest data/mission.json --output-dir outputs/start_area
```

Manifest; RGB görüntüsünü, yükseklik haritasını, izinli hareket alanını, başlangıç bölgesini, rover profilini ve görev noktalarını tanımlar. Rasterların boyut, CRS ve piksel dönüşümü eşleşmelidir. Yatay ve düşey değerler metre cinsinden kullanılır.

| Seçenek | İşlev |
| --- | --- |
| `select_within_zone` | İzinli başlangıç bölgesinden aday seçer |
| `assess_fixed_start` | Verilen başlangıç noktasını değiştirmeden değerlendirir |
| `ground_mode: reference` | Sağlanan zemin maskesini kullanır |
| `ground_mode: predicted` | Yerel yüzey modelinden zemin maskesi üretir |
| `elevation_quality.mode: nominal` | Yükseklik göstergelerini doğrudan kullanır |
| `elevation_quality.mode: bounded_error` | Sağlanan mutlak hata sınırıyla hesaplanan üst sınırları kullanır |

Rover profili en, boy, çevre payı, eğim ve yüzey sınırlarını içerir. Hata sınırı bir güven yüzdesi değildir. Kullanılan araca ve veriye göre ayarlanır.

### Potsdam örneği

Kaynak dosyaları hazırladıktan sonra:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_start_area_demo.py
.\.venv\Scripts\python.exe main.py start-area --manifest data/datasets/potsdam/start_demo/manifest.json --output-dir outputs/potsdam
```

Bu örnek, hazır zemin etiketlerini kullanır. Otomatik zemin ayrımı için `ground_mode: predicted` ve eğitilmiş bir yüzey modeli seçilir.

Metre tabanlı akış `result.json`, `selection.png`, `maps.npz`, `perception.npz` ve `elevation_quality.npz` dosyalarını üretir. JSON çıktısı karar durumunu, seçilen noktayı, rotayı, kullanılan ayarları ve nokta bazındaki eleme nedenlerini içerir.

## Testler

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-surface.txt -r requirements-geospatial.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Bağımlılık sürümlerini sabitlemek için ilgili `requirements-*-lock.txt` dosyaları kullanılabilir. Çalıştırma seçeneklerinin tamamı için `main.py --help` ve `main.py start-area --help` komutlarına bakın.
