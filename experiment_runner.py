import argparse
import numpy as np

from src.models import evet_hayir_modeli_olustur
from src.models import gelismis_topluluk_modeli_olustur
from src.models import duz_cizgi_modeli_olustur


def sahte_banka_verisi_uret():
    """
    simülasyon için sentetik bankacılık veri seti üretir.
    %95 normal işlem, %5 sahtekarlık (fraud) oranı ile dengesiz dağılımı simüle eder
    ve sahtekarlık işlemlerine gerçekçi davranışsal sinyaller enjekte eder.
    """
    np.random.seed(42)
    toplam_islem = 1000
    ozellik_sayisi = 10
    
    X = np.random.randn(toplam_islem, ozellik_sayisi)
    y = np.random.choice([0, 1], size=toplam_islem, p=[0.95, 0.05])
    
    # sahtekarlık (y == 1) işlemlerine anlamlı davranışsal sapmalar ekleniyor
    fraud_indeksleri = np.where(y == 1)[0]
    
    # 1. ve 2. özellik (ör. işlem tutarı ve işlem hızı): fraud işlemlerde ortalaması yüksek
    X[fraud_indeksleri, 0] += 2.2
    X[fraud_indeksleri, 1] += 1.8
    
    # 3. özellik (ör. hesap yaşı veya lokasyon tutarlılığı): fraud işlemlerde ortalaması düşük
    X[fraud_indeksleri, 2] -= 1.5
    
    return X, y


def main():

    # -------------------------------------------------------------------------
    # FAZ 1: Altyapı ve Komut Satırı Argüman Yönetimi
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Asimetrik Finansal Risk ve Model Kararlılığı Test Platformu",
        formatter_class=argparse.RawTextHelpFormatter
    )
    args = parser.parse_args()
     
    print("[INFO] Benchmark motoru basariyla baslatildi...")
    print("[INFO] Platform mimarisi: 3 farkli makine ogrenmesi modeli es zamanli yaristirilacaktir.")
    print("[INFO] 1000 adet zaman sirali islem yuklendi.")
    
    X, y = sahte_banka_verisi_uret()
    
    # zaman duvarı: Veri sızıntısını (data leakage) önlemek için kronolojik bölme
    X_train, X_test = X[:750], X[750:]
    y_train, y_test = y[:750], y[750:]
    print("[INFO] Zaman duvari cekildi! Egitim verisi: 750, Test verisi: 250\n")
    
    # jilet Refactor: Model fabrikasından gelen fonksiyon adresleri sözlüğe bağlanıyor
    modeller_havuzu = {
        "DT": {"isim": "Decision Tree (Kural Tabanli Agac Modeli)", "fonksiyon": evet_hayir_modeli_olustur},
        "RF": {"isim": "Random Forest (Gelismis Topluluk Modeli)", "fonksiyon": gelismis_topluluk_modeli_olustur},
        "LR": {"isim": "Logistic Regression (Dogrusal Parametrik Model)", "fonksiyon": duz_cizgi_modeli_olustur}
    }
    
    model_nihai_zararlari = {}
    model_nihai_kararsizliklari = {}

    # -------------------------------------------------------------------------
    # FAZ 2: Monte Carlo Simülasyonu ve Bootstrap Veri Sarsıntısı
    # -------------------------------------------------------------------------
    for model_kod, model_bilgi in modeller_havuzu.items():
        model_isim = model_bilgi["isim"]
        model_fonksiyonu = model_bilgi["fonksiyon"]
        
        print(f"[!] {model_isim} icin 50 turluk Monte Carlo stres testi baslatildi...")
        tum_turlar_tahmin_olasiliklari = []
        
        for sarsinti_seed in range(1, 51):
            rng = np.random.default_rng(seed=sarsinti_seed)
            
            # bootstrap Resampling yöntemiyle eğitim veri dağılımı sarsılıyor
            secilen_indeksler = rng.choice(750, size=750, replace=True)
            X_train_sarsilmis = X_train[secilen_indeksler]
            y_train_sarsilmis = y_train[secilen_indeksler]
            
            # dinamik model üretimi: Hantal if/elif blokları tamamen ortadan kaldırıldı
            model = model_fonksiyonu(sarsinti_seed)

            model.fit(X_train_sarsilmis, y_train_sarsilmis)
            
            o_turun_tahminleri = model.predict_proba(X_test)[:, 1]
            tum_turlar_tahmin_olasiliklari.append(o_turun_tahminleri)

        # -------------------------------------------------------------------------
        # FAZ 3: Selection Entropy (Seçim Kararsızlığı) Hesaplama Motoru
        # -------------------------------------------------------------------------
        matris_tahminler = np.array(tum_turlar_tahmin_olasiliklari)
        ortalama_p = np.mean(matris_tahminler, axis=0)
        ortalama_p = np.clip(ortalama_p, 1e-15, 1 - 1e-15)
        
        secim_entropisi = - (ortalama_p * np.log2(ortalama_p) + (1 - ortalama_p) * np.log2(1 - ortalama_p))
        genel_model_kararsizligi = np.mean(secim_entropisi)
        
        model_nihai_kararsizliklari[model_kod] = genel_model_kararsizligi

        # -------------------------------------------------------------------------
        # FAZ 4: Cost-Sensitive Evaluation (Asimetrik Maliyet Süzgeci)
        # -------------------------------------------------------------------------
        c_fn = 20  # False Negative Maliyeti
        c_fp = 1   # False Positive Maliyeti
        
        finansal_esik = c_fp / (c_fp + c_fn)
        finansal_kararlar = (ortalama_p >= finansal_esik).astype(int)

        gercek_hirsizlar = y_test.astype(int)
        yanlis_alarm = np.sum((finansal_kararlar == 1) & (gercek_hirsizlar == 0))
        kacirilan_hirsiz = np.sum((finansal_kararlar == 0) & (gercek_hirsizlar == 1))
        toplam_finansal_zarar = (yanlis_alarm * c_fp) + (kacirilan_hirsiz * c_fn)
        
        model_nihai_zararlari[model_kod] = toplam_finansal_zarar
        
        print(f"    [+] Matematiksel Optimal Esik: {finansal_esik:.4f}")
        print(f"    [+] Bloke Edilen Musteri: {yanlis_alarm} | Kacirilan Sahtekarlik: {kacirilan_hirsiz}")
        print(f"    [+] Net Finansal Zarar: {toplam_finansal_zarar} USD\n")

    # -------------------------------------------------------------------------
    # FAZ 5: Karşılaştırmalı Analiz ve Dinamik Model Seçimi
    # -------------------------------------------------------------------------
    print("="*75)
    print("      MODELLER ARASI RESMİ FİNANSAL KIYASLAMA VE BENCHMARK RAPORU")
    print("="*75)
    
    for model_kod, model_bilgi in modeller_havuzu.items():
        print(f"MODEL: {model_kod:<4} | Secim Kararsizligi (Entropy): {model_nihai_kararsizliklari[model_kod]:.4f} | Beklenen Finansal Zarar: {model_nihai_zararlari[model_kod]:>5} USD")
    
    lider_model_kod = min(model_nihai_zararlari, key=model_nihai_zararlari.get)
    
    print("\n[ANALİZ SONUCU VE DİNAMİK MODEL SEÇİMİ]")
    print("    Asimetrik maliyet süzgecinden basariyla gecen ve")
    print("    risk bilancosunu en optimize düzeyde tutan lider model:")
    print(f"     SECILEN LIDER MODEL: {modeller_havuzu[lider_model_kod]['isim']}")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
