# Matematik Mühendisliği Lisans Bitirme Tezi Projesi

Bu proje, Yıldız Teknik Üniversitesi Matematik Mühendisliği bölümü lisans bitirme tezi kapsamında geliştirilen bir nümerik kök bulma yöntemleri projesinin implementasyonlarını ve gerekli dokümanlarını içermektedir. Amaç, temiz, test edilebilir ve matematiksel olarak doğru kodlar üreterek akademik bir standardı korumaktır.

Serhet Gökdemir tarafından, Hale Gonce Köçken danışmanlığında geliştirilmiştir.

## Proje Yapısı

Proje, hem tez metnini (LaTeX), hem sayısal yöntemlerin Python implementasyonlarını, hem de deney/sonuç dosyalarını içerecek şekilde aşağıdaki hiyerarşi etrafında organize edilmiştir. Bazı klasörler henüz tamamen dolu olmayabilir; yapı hedef mimariyi göstermektedir.

```
thesis-nonlinear-equations/
│
├── README.md
├── .gitignore
│
├── thesis/
│   ├── main.tex
│   │
│   ├── chapters/
│   │   ├── introduction.tex
│   │   ├── mathematical_background.tex
│   │   ├── single_nonlinear_equations.tex
│   │   ├── systems_of_nonlinear_equations.tex
│   │   ├── implementation_and_experiments.tex
│   │   ├── optimization_connection.tex
│   │   └── conclusion.tex
│   │
│   ├── frontmatter/
│   │   ├── abstract.tex
│   │   ├── preface.tex
│   │   ├── symbols.tex
│   │   ├── abbreviations.tex
│   │   └── report_sections.tex   <-- (NEW)
│   │
│   ├── figures/
│   │   ├── single_variable/
│   │   ├── systems/
│   │   └── experiments/
│   │
│   ├── tables/
│   │   └── generated/           <-- (NEW, latex table çıktıları)
│   │
│   ├── bibliography/
│   │   └── references.bib
│   │
│   └── output/
│       └── main.pdf
│
├── src/
│   ├── single_variable/
│   │   ├── bisection.py
│   │   ├── secant.py
│   │   ├── newton.py
│   │   ├── damped_newton.py
│   │   └── brent.py
│   │
│   ├── systems/
│   │   ├── newton_system.py
│   │   └── broyden.py
│   │
│   ├── utils/
│   │   ├── stopping_criteria.py
│   │   ├── norms.py
│   │   ├── derivatives.py
│   │   ├── jacobian.py
│   │   └── plotting.py
│   │
│   └── __init__.py
│
├── experiments/
│   ├── single_variable_experiments.py
│   ├── system_experiments.py
│   ├── generate_tables.py        <-- (NEW)
│   └── generate_plots.py         <-- (NEW)
│
├── tests/
│   ├── test_bisection.py
│   ├── test_secant.py
│   ├── test_newton.py
│   ├── test_damped_newton.py
│   ├── test_brent.py
│   ├── test_newton_system.py
│   └── test_broyden.py
│
├── data/
│   ├── raw/
│   └── generated_results/
│       ├── csv/
│       └── tables/
│
└── notebooks/
	├── single_variable_demo.ipynb
	└── systems_demo.ipynb

# Mevcut ekstra klasörler
- checkpoints/ : Tarihsel ilerleme PDF’leri (ör. 03.17.2026.pdf)
- history/     : Geçmiş/dökümantasyon için yer tutucu
- forbidden/   : Kişisel veya repoya dahil edilmeyen dosyalar (gitignore)
- venv/        : Sanal ortam (gitignore)
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
