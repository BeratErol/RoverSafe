# Veri kaynakları

| Kaynak | Kullanım |
| --- | --- |
| [AeroScapes](https://github.com/ishann/aeroscapes) | Hava görüntüleri ve yüzey sınıflandırma maskeleri |
| [Kaggle Drone Dataset](https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset) | Görüntü ve maskelerle yöntem karşılaştırması |
| [ISPRS Potsdam](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx) | Eşleşen ortofoto, yükseklik haritası ve sınıf etiketleri |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | Önceden eğitilmiş nesne tespit modeli |

Potsdam verisinin kaynak atfı ISPRS / BSF Swissphoto'dur. AeroScapes indirme araçları [Hugging Face aynasını](https://huggingface.co/datasets/dronefreak/Aeroscapes) kullanır; özgün kaynak bağlantısı manifestlere kaydedilir. Her kaynağın kullanım koşulları kendi sayfasında yer alır.

## Veri hazırlama

- `scripts/download_aeroscapes.py`: küçük AeroScapes alt kümesi
- `scripts/download_aeroscapes_full.py`: AeroScapes veri hazırlığı
- `scripts/download_kaggle_drone.py`: Kaggle görüntü ve maskeleri
- `scripts/download_potsdam_sample.py`: Potsdam örnek karosu
- `scripts/download_potsdam_tiles.py`: seçilen Potsdam karoları

Potsdam araçları resmî paylaşımın doğrudan arşiv bağlantısını `--archive-url` ile alır. Gerekli bağımlılıklar `requirements-data.txt` içindedir.

Görüntüler, videolar ve büyük raster dosyaları `data/datasets/`, `data/images/` ve `data/videos/` altında yerel olarak tutulur. Depoda kaynak bilgileri, manifestler ve sentetik örnek bulunur. Sınıf etiketleri, fiziksel geçilebilirlik ölçümü değildir.
