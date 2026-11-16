import argparse
import numpy as np

from src.models import evet_hayir_modeli_olustur
from src.models import gelismis_topluluk_modeli_olustur
from src.models import duz_cizgi_modeli_olustur


def sahte_banka_verisi_uret():
    """
    simulasyon icin zamansal kayma (concept drift) iceren sentetik bankacilik verisi uretir.
    %95 normal islem, %5 sahtekarlik (fraud) orani ile dengesiz dagilimi simule eder.
    """
    rng = np.random.default_rng(seed=42)
    toplam_islem = 1000
    ozellik_sayisi = 10

    x = rng.standard_normal((toplam_islem, ozellik_sayisi))
    y = rng.choice(list((0, 1)), size=toplam_islem, p=list((0.95, 0.05)))

    # zaman trendi (concept drift): zaman ilerledikce islem davranislari hafifce kayar.
    zaman_trendi = np.linspace(0.0, 0.6, toplam_islem)
    x[:, 0] += zaman_trendi
    x[:, 1] -= zaman_trendi * 0.5

    # sahtekarlik (y == 1) islemlerine davranissal sapmalar enjekte ediliyor
    fraud_indeksleri = np.where(y == 1)[0]

    # 1. ve 2. ozellik (islem tutari ve hizi): fraud islemlerde yuksek
    x[fraud_indeksleri, 0] += 2.2
    x[fraud_indeksleri, 1] += 1.8

    # 3. ozellik (hesap yasi / lokasyon tutarliligi): fraud islemlerde dusuk
    x[fraud_indeksleri, 2] -= 1.5

    return x, y


def ikili_entropi_hesapla(olasilik_dizisi):
    """shannon ikili entropi formulu: h(p) = -p*log2(p) - (1-p)*log2(1-p)"""
    p = np.clip(olasilik_dizisi, 1e-15, 1.0 - 1e-15)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def main():
    # -------------------------------------------------------------------------
    # faz 1: altyapi ve komut satiri arguman yonetimi
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="asimetrik finansal risk ve model kararliligi test platformu (final)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--tur", type=int, default=50, help="bootstrap tur sayisi (varsayilan: 50)")
    parser.add_argument("--fn-maliyet", type=float, default=20.0, help="sahtekarligi kacirma maliyeti (varsayilan: 20.0)")
    parser.add_argument("--fp-maliyet", type=float, default=1.0, help="yanlis alarm maliyeti (varsayilan: 1.0)")
    parser.add_argument("--kararsizlik-cezasi", type=float, default=50.0, help="kararsizlik ceza katsayisi (varsayilan: 50.0)")
    args = parser.parse_args()

    print("[INFO] Benchmark motoru başarıyla başlatıldı...")
    print(f"[INFO] Parametreler -> Tur: {args.tur} | FN: {args.fn_maliyet} | FP: {args.fp_maliyet}")

    x, y = sahte_banka_verisi_uret()

    # zaman duvari: veri sizintisini (data leakage) onlemek icin kronolojik bolme
    x_train, x_test = x[:750], x[750:]
    y_train, y_test = y[:750], y[750:]
    print("[INFO] Zaman duvarı çekildi! Eğitim verisi: 750, Test verisi: 250\n")

    modeller_havuzu = {
        "dt": {"isim": "Decision Tree (Kural Tabanlı Ağaç Modeli)", "fonksiyon": evet_hayir_modeli_olustur},
        "rf": {"isim": "Random Forest (Gelişmiş Topluluk Modeli)", "fonksiyon": gelismis_topluluk_modeli_olustur},
        "lr": {"isim": "Logistic Regression (Doğrusal Parametrik Model)", "fonksiyon": duz_cizgi_modeli_olustur},
    }

    model_nihai_zararlari = {}
    model_nihai_kararsizliklari = {}
    model_bilesik_skorlari = {}

    # -------------------------------------------------------------------------
    # faz 2: monte carlo simulasyonu ve bootstrap veri sarsintisi
    # -------------------------------------------------------------------------
    for model_kod, model_bilgi in modeller_havuzu.items():
        model_isim = model_bilgi["isim"]
        model_fonksiyonu = model_bilgi["fonksiyon"]

        print(f"[!] {model_isim} için {args.tur} turluk stres testi başlatıldı...")
        tum_turlar_tahmin_olasiliklari = []

        for sarsinti_seed in range(1, args.tur + 1):
            rng = np.random.default_rng(seed=sarsinti_seed)

            secilen_indeksler = rng.choice(len(x_train), size=len(x_train), replace=True)
            x_train_sarsilmis = x_train[secilen_indeksler]
            y_train_sarsilmis = y_train[secilen_indeksler]

            model = model_fonksiyonu(sarsinti_seed)
            model.fit(x_train_sarsilmis, y_train_sarsilmis)

            o_turun_tahminleri = model.predict_proba(x_test)[:, 1]
            tum_turlar_tahmin_olasiliklari.append(o_turun_tahminleri)

        # -------------------------------------------------------------------------
        # faz 3: epistemik belirsizlik (tur-ici gercek model kararsizligi)
        # -------------------------------------------------------------------------
        matris_tahminler = np.array(tum_turlar_tahmin_olasiliklari)
        ortalama_p = np.mean(matris_tahminler, axis=0)

        toplam_entropi = ikili_entropi_hesapla(ortalama_p)
        tur_ici_ortalama_entropi = np.mean(ikili_entropi_hesapla(matris_tahminler), axis=0)
        epistemik_kararsizlik = np.maximum(0.0, toplam_entropi - tur_ici_ortalama_entropi)

        genel_model_kararsizligi = float(np.mean(epistemik_kararsizlik))
        model_nihai_kararsizliklari[model_kod] = genel_model_kararsizligi

        # -------------------------------------------------------------------------
        # faz 4: cost-sensitive evaluation (asimetrik maliyet suzgeci)
        # -------------------------------------------------------------------------
        c_fn = args.fn_maliyet
        c_fp = args.fp_maliyet

        finansal_esik = c_fp / (c_fp + c_fn)
        finansal_kararlar = (ortalama_p >= finansal_esik).astype(int)

        gercek_hirsizlar = y_test.astype(int)
        yanlis_alarm = int(np.sum((finansal_kararlar == 1) & (gercek_hirsizlar == 0)))
        kacirilan_hirsiz = int(np.sum((finansal_kararlar == 0) & (gercek_hirsizlar == 1)))
        toplam_finansal_zarar = (yanlis_alarm * c_fp) + (kacirilan_hirsiz * c_fn)

        bilesik_risk_skoru = toplam_finansal_zarar + (genel_model_kararsizligi * args.kararsizlik_cezasi)

        model_nihai_zararlari[model_kod] = toplam_finansal_zarar
        model_bilesik_skorlari[model_kod] = bilesik_risk_skoru

        print(f"    [+] Optimal Eşik: {finansal_esik:.4f} | FP: {yanlis_alarm} | FN: {kacirilan_hirsiz}")
        print(f"    [+] Finansal Zarar: {toplam_finansal_zarar:.1f} USD | Epistemik Kararsızlık: {genel_model_kararsizligi:.4f}\n")

    # -------------------------------------------------------------------------
    # faz 5: karsilastirmali analiz ve risk-dengeli lider secimi
    # -------------------------------------------------------------------------
    print("=" * 88)
    print("           MODELLER ARASI RESMİ FİNANSAL KIYASLAMA VE BENCHMARK RAPORU")
    print("=" * 88)

    for model_kod in modeller_havuzu:
        print(
            f"MODEL: {model_kod.upper():<4} | "
            f"Epistemik Kararsızlık: {model_nihai_kararsizliklari[model_kod]:.4f} | "
            f"Finansal Zarar: {model_nihai_zararlari[model_kod]:>6.1f} USD | "
            f"Bileşik Risk Skoru: {model_bilesik_skorlari[model_kod]:>6.2f}"
        )

    lider_model_kod = min(model_bilesik_skorlari, key=model_bilesik_skorlari.get)

    print("\n[ANALİZ SONUCU VE DİNAMİK MODEL SEÇİMİ]")
    print(f"     SEÇİLEN LİDER MODEL: {modeller_havuzu[lider_model_kod]['isim']}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
