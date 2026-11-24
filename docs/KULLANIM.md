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

## Referans rover profili

`configs/rover_husky_a200.json` yayımlanmış Husky A200 boyutlarını ve proje varsayımlarını ayırır. Metre tabanlı görevde dosyanın `profile` nesnesini manifestin `rover` alanına, kaynak bilgisini kaydetmek için `provenance` nesnesini `rover_profile_provenance` alanına kopyalayın. Eğim ve hareket sınırları sahada ölçülmemiştir.

## Arazi maliyetine göre rota seçimi

Metre tabanlı görev manifestine aşağıdaki bölüm eklenebilir:

```json
"routing": {
  "mode": "terrain_cost",
  "terrain_weight": 4.0,
  "clearance_weight": 1.0
}
```

Varsayılan `mode: shortest` geometrik uzunluğu en aza indirir. `terrain_cost`, zorunlu koşulları geçen hücrelerde eğim/pürüzlülük/yerel yükseklik şiddetini ve engel yakınlığını da dikkate alır. Ağırlıklar 0–100 arasında sonlu sayılardır; saha kalibrasyonu değildir. İki ağırlık da sıfırsa geometrik mesafe maliyeti kalır. Başlangıç alanı sıralaması değişmez; rota seçilen noktadan hesaplanır.

JSON'daki `route_cost_m_equivalent` mesafe ağırlıklı sezgisel maliyettir; gerçek enerji veya emniyet olasılığı değildir. `route_max_terrain_risk` ve `route_mean_terrain_risk` aynı sezgisel arazi şiddetini özetler. Daha düşük maliyet daha kısa rota anlamına gelmez. Maliyet karşılaştırırken aynı ağırlıklar kullanılmalıdır.

`maps.npz` içinde `terrain_risk` ve `route_cost` haritaları bulunur. `traversability.tif`, kaynak CRS/ızgarada iki bant içerir: `terrain_severity` ve `cost_per_meter`; uygun hareket merkezleri dışındaki nodata değeri -9999'dur.

## Başlangıç ile çıkış arasında asgari mesafe

Metre tabanlı manifestin `mission` bölümüne `"min_start_exit_distance_m": 5.0` eklenebilir. Bir `exit_pixel` gerekir. Değer başlangıç ve çıkış arasındaki kuş uçuşu mesafenin alt sınırıdır; rota uzunluğunu eniyileyen bir hedef değildir. Varsayılan `0.0`, mevcut seçimi korur. Eğim, engel ve rover açıklığı koşulları değişmez. Koşulu sağlayan başlangıç bulunamazsa kısa rotaya dönülmez.

| Karar | Anlamı |
| --- | --- |
| `no_start_meets_exit_distance` | Çıkışa bağlı fiziksel adaylar var; hiçbiri mesafe koşulunu sağlamıyor. |
| `fixed_start_too_close_to_exit` | Sabit başlangıç fiziksel koşulları geçiyor ancak çıkışa fazla yakın. Nokta taşınmaz. |

`best.distance_to_exit_m` kuş uçuşu mesafeyi, `route_length_m` bulunan rotayı gösterir. `maps.npz` içindeki `reachable` fiziksel bağlantıyı korur; `eligible_start` ayrıca asgari mesafeyi uygular. Görseldeki yeşil alanlar ve `decision.tif` sınıf 3 uygun başlangıç merkezleridir.


## Video performans ölçümü

Aynı klip üzerinde temel görüntü işleme hattı ile bir yüzey modeli karşılaştırılabilir:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_video.py --input data/videos/aeroscapes_demo.avi --surface-config configs/surface_dense_urban.json --output-dir outputs/video_benchmark --frames 24 --repeats 3 --strides 1 3
```

Video ve model ağırlıkları önceden hazırlanmalıdır. AeroScapes demo manifesti ve görüntüleri mevcutsa kaynak klip `scripts/create_dataset_video.py` ile üretilebilir. Bu klibin 10 FPS zamanlaması yapaydır; gerçek uçuş kare aralığı değildir. Başka bir yerel video da kullanılabilir. Temel hattı tek başına ölçmek için `--surface-config` kaldırılır. Yeni veya boş çıktı klasörü gerekir.

Araç ilk kareleri en fazla 640 piksel genişliğe, en/boy oranını koruyarak indirir ve bütün profillere aynı MJPG klibi verir. `--width`, `--frames`, `--repeats` ve `--strides` ölçüm kapsamını belirler. Her tekrarda model yeniden yüklenir.

- `summary.json`: giriş SHA-256, çözünürlük, ortam ve paket sürümleri, model/kodlayıcı özetleri, eşik, ilk kare/medyan/P95 süreleri.
- İşleme FPS: kare analizinin hızı; model kurulumunu, video okuma/yazma ve görselleştirmeyi içermez.
- Uçtan uca FPS: model kurulumu, video okuma/yazma ve görselleştirme dahil hız.
- `report.html`: tekrarların ölçümleri ve her çalışmanın görsel/video bağlantıları.

Çıktı videoları sonuna kadar yeniden okunur; kare indeksleri seçilen adımla karşılaştırılır. Kare atlamak tek kare çıkarım maliyetini azaltmaz. Farklı adımlar farklı içerikte kare seçtiğinden hız farkı yalnız adım parametresine bağlanmaz. Kısa klip ölçümü gerçek zaman garantisi değildir.

Bu araç kayıtlı RGB karelerinde görsel ön eleme yapar. Dünya koordinatında takip/SLAM, video üzerinden metrik rota veya hareket kontrolü uygulamaz. YOLO bu karşılaştırmaya varsayılan olarak dahil değildir; bu deney algılama doğruluğunu ölçmez.
