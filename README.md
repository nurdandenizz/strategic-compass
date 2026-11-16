# strategic-compass

a lightweight simulation and benchmarking tool for imbalanced fraud detection under temporal concept drift. instead of standard accuracy metrics, it evaluates models based on epistemic instability (mutual information), asymmetric financial cost (bayes minimum risk), and composite risk scores.

## how it works

- chronological split with concept drift: injects a gradual temporal trend into 1,000 synthetic transactions and splits them strictly by time (750 train / 250 test) to prevent data leakage.
- model pool (`src/models.py`): compares logistic regression, decision tree, and random forest using isolated factory functions.
- 50-round stress test: uses bootstrap resampling across 50 iterations to measure true inter-round epistemic instability (`total entropy - mean round entropy`).
- asymmetric cost filter: replaces the default `0.50` threshold with the bayes minimum risk threshold (`c_fp / (c_fp + c_fn)`). with a 1:20 cost ratio (fp = \$1, fn = \$20), the threshold is set to `0.0476`.
- risk-weighted selection: combines net financial loss and instability penalty into a composite risk score to dynamically select the champion model.

## structure

```text
strategic-compass/
├── data/
├── src/
│   ├── __init__.py
│   └── models.py
├── experiment_runner.py
├── requirements.txt
└── README.md
```

## benchmark output

results from a 50-round run with a 1:20 cost ratio (threshold = `0.0476`, instability penalty = `50.0`):

| model | epistemic instability | false positives | missed frauds | net loss | composite risk |
| :--- | :---: | :---: | :---: | :---: | :---: |
| logistic regression | 0.0090 | 29 | 0 | \$29.0 | 29.45 |
| decision tree | 0.0934 | 33 | 1 | \$53.0 | 57.67 |
| random forest | 0.0118 | 40 | 1 | \$60.0 | 60.59 |

logistic regression achieved both the lowest epistemic instability (0.0090) and the lowest net financial loss (\$29.0) under temporal concept drift, making it the selected champion model.

## usage

```bash
pip install -r requirements.txt
python experiment_runner.py --rounds 50 --fn-cost 20 --fp-cost 1
```

---

# strategic-compass (turkce)

zamansal kayma (concept drift) iceren dengesiz finansal sahtekarlik (fraud) tespiti icin hafif bir simulasyon ve kiyaslama aracidir. standart dogruluk metrikleri yerine, modelleri epistemik kararsizlik (mutual information), asimetrik finansal maliyet (bayes minimum risk) ve bilesik risk skoru uzerinden degerlendirir.

## nasil calisir

- zamansal kaymali kronolojik bolme: 1.000 sentetik isleme kademeli bir zaman trendi ekler ve veri sizintisini onlemek icin zamana gore kesin olarak boler (750 egitim / 250 test).
- model havuzu (`src/models.py`): logistic regression, decision tree ve random forest modellerini izole fabrika fonksiyonlari kullanarak kiyaslar.
- 50 turluk stres testi: turlar arasi gercek epistemik kararsizligi (`toplam entropi - tur ici ortalama entropi`) olcmek icin 50 iterasyon boyunca bootstrap ornekleme kullanir.
- asimetrik maliyet suzgeci: varsayilan `0.50` esigini bayes minimum risk esigi (`c_fp / (c_fp + c_fn)`) ile degistirir. 1:20 maliyet oraninda (fp = \$1, fn = \$20) esik degeri `0.0476` olarak ayarlanir.
- risk dengeli secim: net finansal zarar ile kararsizlik cezasini tek bir bilesik risk skorunda birlestirerek lider modeli dinamik olarak secer.

## yapi

```text
strategic-compass/
├── data/
├── src/
│   ├── __init__.py
│   └── models.py
├── experiment_runner.py
├── requirements.txt
└── README.md
```

## benchmark ciktisi

1:20 maliyet orani ile 50 turluk calistirma sonuclari (esik = `0.0476`, kararsizlik cezasi = `50.0`):

| model | epistemik kararsizlik | yanlis alarmlar | kacirilan sahtekarlik | net zarar | bilesik risk |
| :--- | :---: | :---: | :---: | :---: | :---: |
| logistic regression | 0.0090 | 29 | 0 | $29.0 | 29.45 |
| decision tree | 0.0934 | 33 | 1 | $53.0 | 57.67 |
| random forest | 0.0118 | 40 | 1 | $60.0 | 60.59 |

zamansal kayma altinda logistic regression hem en dusuk epistemik kararsizliga (0.0090) hem de en dusuk net finansal zarara ($29.0) ulasarak lider model olarak secilmistir.

## kullanim

```bash
pip install -r requirements.txt
python experiment_runner.py --rounds 50 --fn-cost 20 --fp-cost 1
```
