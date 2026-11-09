# Yüzey modeli eğitimi ve yöntem karşılaştırması

## Modeller

Renk/doku modeli 14 özellik, çok ölçekli sürüm 35 özellik kullanır. Bağlam modeli dondurulmuş ResNet18 kodlayıcısıyla 419 özellik çıkarır. Yoğun hava modeli aynı özellikler üzerinde evrişimli bölümleme başlığı eğitir. Büyük görüntüler 512 piksel pencere ve 64 piksel örtüşmeyle işlenir; özellik ızgarası tüm görüntüde hizalıdır.

Bakış açısı aerial veya ground olarak kaydedilir; yer kamerası modeli hava modeli gibi kullanılamaz. Tahmin skoru fiziksel emniyet olasılığı değildir. Güvenilir yerel joblib modelleri ve weights_only=True ile okunan yoğun model paketleri kullanılır.

## Kurulum ve tekrar üretim

Python 3.12 ortamında:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-context.txt -r requirements-data.txt
.\.venv\Scripts\python.exe scripts/download_context_encoder.py
.\.venv\Scripts\python.exe scripts/prepare_aerial_protocol.py
.\.venv\Scripts\python.exe scripts/train_dense_aerial.py --protocol outputs/dense_expanded_protocol.json --output-dir outputs/dense_aerial_full --epochs 24
```

Potsdam geliştirme karoları 2_11, 3_11, 4_10, 5_10; doğrulama 4_11; test 6_11 olmalıdır. İlgili RGB ve etiket karoları data/datasets/potsdam altında önceden hazırlanır. Protokol bu dosyaları ve özetlerini kaydeder. Yeni deney için yeni çıktı klasörü ve protokol yolu kullanılır.

Geliştirme verisiyle eğitim yapılır; model/eşik yalnız doğrulamada seçilir. Precision ≥ 0,80 ve recall ≥ 0,40 koşulunu sağlayan adaylar arasında F1 en büyüklenir. Koşul sağlanmazsa başarısızlık kaydedilir. Test, selection.json yazıldıktan sonra okunur; test sonucu eşik ayarında kullanılmaz. Daha önce incelenmiş test karosu regresyon karşılaştırmasıdır; bağımsız doğal saha testi sayılmaz.

Önceki bağlam modelini yeniden üretmek için:

```powershell
.\.venv\Scripts\python.exe scripts/train_context_surface.py --manifest outputs/dense_expanded_protocol.json --domain aerial --output-dir outputs/context_aerial
.\.venv\Scripts\python.exe scripts/compare_aerial_models.py --protocol outputs/dense_expanded_protocol.json --baseline outputs/context_aerial/selection.json --improved outputs/dense_aerial_full/selection.json --output-dir outputs/aerial_urban_comparison
```

Her modelin kendi selection.json dosyası gerekir; karşılaştırıcı kayıtlı model özeti ve eşiğini kullanır. Birleşik doğal arazi modeli ve dış test araçları bu paketin parçası değildir.

## Kaydedilmiş kentsel deney

| Aynı 6_11 karosunda yöntem | Precision | Recall | IoU |
| --- | ---: | ---: | ---: |
| Önceki bağlam modeli | %72,49 | %56,22 | %46,33 |
| Yoğun kentsel model | %89,76 | %54,58 | %51,38 |

Karşılaştırma aynı 49 test penceresinde 20 piksel kenar dışlamasıyla yapılmıştır. Mimari ve geliştirme verisi birlikte değiştiğinden fark yalnız mimari etkisi sayılmaz. Precision yükselmiş, recall düşmüştür. Semantik hedef sınıfı fiziksel sürülebilirlik etiketi değildir.

Model ağırlıkları, kodlayıcı ve büyük veriler depoda bulunmaz. Yerel kayıtlarda plan.json, training.json, selection.json, test.json ve karşılaştırma summary.json/report.html bulunur. configs/surface_dense_urban.json bu yerel ağırlıklar hazırlandıktan sonra kullanılır:

```powershell
.\.venv\Scripts\python.exe main.py data/samples/synthetic_terrain.png --config configs/surface_dense_urban.json --output-dir outputs/model_smoke
```

Sentetik görüntü komutu yalnız çalıştırma örneğidir, doğruluk ölçümü değildir. Gerçek görüntüde aynı komuta uygun görüntü yolu verilir.
