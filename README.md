# strategic-compass

a lightweight simulation and benchmarking tool for imbalanced fraud detection. instead of standard accuracy metrics, it evaluates models based on prediction stability (selection entropy) and asymmetric financial cost (bayes minimum risk).

## how it works

- chronological split: splits the 1,000 synthetic transactions strictly by time (750 train / 250 test) to avoid future data leakage.
- model pool (`src/models.py`): compares logistic regression, decision tree, and random forest using isolated factory functions.
- 50-round stress test: uses bootstrap resampling across 50 iterations to measure how much each model's predictions fluctuate (selection entropy).
- asymmetric cost filter: replaces the default `0.50` threshold with the bayes minimum risk threshold (`c_fp / (c_fp + c_fn)`). with a 1:20 cost ratio (fp = \$1, fn = \$20), the threshold is set to `0.0476`.

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

results from a 50-round run with a 1:20 cost ratio (threshold = `0.0476`):

| model | selection entropy | false positives | missed frauds | net loss |
| :--- | :---: | :---: | :---: | :---: |
| decision tree | 0.0729 | 14 | 1 | \$34 |
| logistic regression | 0.0701 | 18 | 1 | \$38 |
| random forest | 0.1435 | 24 | 1 | \$44 |

logistic regression had the lowest prediction variance (0.0701), while decision tree resulted in the lowest total financial loss (\$34) due to fewer false customer blocks.

## usage

```bash
pip install -r requirements.txt
python experiment_runner.py
```

---

# strategic-compass (turkce)

dengesiz dagilima sahip finansal sahtekarlik (fraud) tespiti icin hafif bir simulasyon ve kiyaslama aracidir. standart dogruluk metrikleri yerine, modelleri tahmin kararliligi (selection entropy) ve asimetrik finansal maliyet (bayes minimum risk) uzerinden degerlendirir.

## nasil calisir

- kronolojik bolme: gelecekteki verinin gecmise sizmasini onlemek icin 1.000 sentetik islemi zamana gore kesin olarak boler (750 egitim / 250 test).
- model havuzu (`src/models.py`): logistic regression, decision tree ve random forest modellerini izole fabrika fonksiyonlari kullanarak kiyaslar.
- 50 turluk stres testi: her modelin tahminlerinin ne kadar dalgalandigini (selection entropy) olcmek icin 50 iterasyon boyunca bootstrap ornekleme kullanir.
- asimetrik maliyet suzgeci: varsayilan `0.50` esigini bayes minimum risk esigi (`c_fp / (c_fp + c_fn)`) ile degistirir. 1:20 maliyet oraninda (fp = \$1, fn = \$20) esik degeri `0.0476` olarak ayarlanir.

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

1:20 maliyet orani ile 50 turluk calistirma sonuclari (esik = `0.0476`):

| model | secim kararsizligi | yanlis alarmlar | kacirilan sahtekarlik | net zarar |
| :--- | :---: | :---: | :---: | :---: |
| decision tree | 0.0729 | 14 | 1 | $34 |
| logistic regression | 0.0701 | 18 | 1 | $38 |
| random forest | 0.1435 | 24 | 1 | $44 |

logistic regression en dusuk tahmin varyansina (0.0701) sahip olurken, decision tree daha az sayida musteri blokesi sayesinde en dusuk toplam finansal zarari ($34) saglamistir.

## kullanim

```bash
pip install -r requirements.txt
python experiment_runner.py
```
