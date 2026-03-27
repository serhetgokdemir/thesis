# Sayısal Analiz Yöntemleri Tez Projesi

Bu proje, sayısal analiz dersi kapsamında geliştirilen ve bir yüksek lisans tezinin parçası olan sayısal kök bulma yöntemlerinin Python implementasyonlarını içermektedir. Amaç, temiz, test edilebilir ve matematiksel olarak doğru kodlar üreterek akademik bir standardı korumaktır.

## Proje Yapısı

Proje, sayısal yöntemlerin kendisini (`src`) ve bu yöntemleri doğrulayan testleri (`tests`) içerecek şekilde iki ana klasöre ayrılmıştır.

```
.
├── src
│   ├── single_variable
│   │   ├── bisection.py
│   │   ├── brent.py
│   │   ├── damped_newton.py
│   │   ├── newton.py
│   │   └── secant.py
│   └── systems
│       ├── broyden.py
│       └── newton_system.py
├── tests
│   ├── test_bisection.py
│   ├── test_brent.py
│   # ... diğer testler
└── requirements.txt
```

## Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### 1. Sanal Ortam Oluşturma (Önerilir)

Proje bağımlılıklarını sisteminizden izole etmek için bir sanal ortam oluşturmanız tavsiye edilir.

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Bağımlılıkları Yükleme

Projenin ihtiyaç duyduğu kütüphaneler `requirements.txt` dosyasında listelenmiştir. Bu bağımlılıkları `pip` ile yükleyebilirsiniz.

```bash
pip install -r requirements.txt
```

### 3. Testleri Çalıştırma

Tüm sayısal yöntemlerin doğruluğunu ve kararlılığını kontrol etmek için `pytest` tabanlı testleri çalıştırın.

```bash
pytest
```

Tüm testler başarılı bir şekilde geçiyorsa, kurulum tamamlanmış ve kodlar doğru çalışıyor demektir.
