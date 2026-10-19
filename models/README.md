# Modeller

## Nesne tespiti

`yolo11n.pt`, Ultralytics'in önceden eğitilmiş nesne tespit modelidir. `scripts/download_yolo_model.py` ile hazırlanır. Bağımlılıkları `requirements-yolo.txt` içindedir.

## Yüzey sınıflandırma

| Dosya | Açıklama | Eğitim aracı |
| --- | --- | --- |
| `surface.joblib` | AeroScapes görüntülerinden eğitilen ExtraTrees modeli | `scripts/train_surface.py` |
| `surface_potsdam.joblib` | Potsdam görüntülerinden eğitilen alternatif yüzey modeli | `scripts/train_potsdam_surface.py` |

Yüzey modelleri renk ve yerel dokudan çıkarılan 14 özellik kullanır. Piksel koordinatları veya referans etiketler tahmin girdisi değildir. Bağımlılıkları `requirements-surface.txt` içindedir.

Modelin yanındaki JSON dosyası eğitim örneklerini, parametreleri, sınıf sayılarını ve dosya özetlerini kaydeder. Model seçimi ve eşik, kullanılan ayar dosyasında belirtilir. Sınıf puanı fiziksel güvenlik olasılığı değildir.

İkili model ağırlıkları depoya eklenmez. Yalnızca güvenilir yerel `joblib` dosyalarını yükleyin. Sabit bağımlılık sürümleri için ilgili `requirements-*-lock.txt` dosyalarını kullanın.
