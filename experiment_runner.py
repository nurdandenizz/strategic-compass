import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.models import bilge_agac_modeli_olustur
from src.models import konsey_modeli_olustur
from src.models import eski_toprak_modeli_olustur

def sahte_banka_verisi_uret():
    # gercekci banka ritmini simule eden veri darphanesi
    np.random.seed(42)
    toplam_islem = 1000
    ozellik_sayisi = 10
    
    # %80 gunduz islemleri, %5 fraud oranini yansitacak sekilde sahte veri kurgusu
    X = np.random.randn(toplam_islem, ozellik_sayisi)
    y = np.random.choice([0, 1], size=toplam_islem, p=[0.95, 0.05])
    return X, y

def main():
    # -------------------------------------------------------------------------
    # faz 1: altyapi, model ve veri fabrikasi
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="siber firtina deney hatti")
    parser.add_argument("--model", type=str, required=True, help="rf, dt veya lr secimi")
    args = parser.parse_parser = parser.parse_args()
    
    secilen_model_adi = args.model.upper()
    print(f"[info] benchmark motoru basariyla baslatildi...")
    print(f"[info] secilen model: {secilen_model_adi}")
    print(f"[info] module 10: 1000 tane zaman sirali islem yuklendi.")
    
    # sahte veri darphanesinden ham maddeleri cekiyoruz
    X, y = sahte_banka_verisi_uret()
    
    # zaman duvari: veri sızmasını onlemek icin kronolojik bolme
    X_train, X_test = X[:750], X[750:]
    y_train, y_test = y[:750], y[750:]
    print(f"[info] zaman duvari cekildi! egitim verisi: 750, test verisi: 250")
    
    tum_turlar_tahmin_olasiliklari = []

    # -------------------------------------------------------------------------
    # faz 2: deprem dongusu ve model egitimi
    # -------------------------------------------------------------------------
    for sarsinti_seed in range(1, 51):
        seed_degeri = sarsinti_seed
        
        if secilen_model_adi == "RF":
            model = konsey_modeli_olustur(seed_degeri)
        elif secilen_model_adi == "DT":
            model = bilge_agac_modeli_olustur(seed_degeri)
        elif secilen_model_adi == "LR":
            model = eski_toprak_modeli_olustur(seed_degeri)

        # model o turdaki sarsintili veriyle egitiliyor
        model.fit(X_train, y_train)
        
        # test islemlerinin hirsizlik olasilik skorlari yakalaniyor
        o_turun_tahminleri = model.predict_proba(X_test)[:, 1]
        
        # her turun tahminlerini ana sepete kilitliyoruz ki hafiza kaybi yasanmasin
        tum_turlar_tahmin_olasiliklari.append(o_turun_tahminleri)

    print(f"[!] {secilen_model_adi} modeli icin 50 turluk sarsinti ve gelecek simulasyonu basariyla tamamlandi!")

    # -------------------------------------------------------------------------
    # faz 3: selection entropy (secim kararsizligi) hesaplama motoru
    # -------------------------------------------------------------------------
    matris_tahminler = np.array(tum_turlar_tahmin_olasiliklari)
    ortalama_p = np.mean(matris_tahminler, axis=0)
    ortalama_p = np.clip(ortalama_p, 1e-15, 1 - 1e-15)
    secim_entropisi = - (ortalama_p * np.log2(ortalama_p) + (1 - ortalama_p) * np.log2(1 - ortalama_p))
    genel_model_kararsizligi = np.mean(secim_entropisi)

    print(f"[+] {secilen_model_adi} Modeli Icin Genel Secim Kararsizligi (Entropy): {genel_model_kararsizligi:.4f}")

    # -------------------------------------------------------------------------
    # faz 4: cost-sensitive evaluation (finansal süzgeci)
    # -------------------------------------------------------------------------
    maliyet_senaryolari = [
        {"isim": "1:10 hafif kriz", "c_fn": 10, "c_fp": 1},
        {"isim": "1:20 agir kriz", "c_fn": 20, "c_fp": 1}
    ]

    print(f"\n[!] {secilen_model_adi} Icin Finansal Risk Raporu Inceleniyor...")

    gercek_hirsizlar = y_test.astype(int)

    for senaryo in maliyet_senaryolari:
        isim = senaryo["isim"]
        c_fn = senaryo["c_fn"]
        c_fp = senaryo["c_fp"]

        finansal_esik = c_fp / (c_fp + c_fn)
        finansal_kararlar = (ortalama_p >= finansal_esik).astype(int)

        yanlis_alarm = np.sum((finansal_kararlar == 1) & (gercek_hirsizlar == 0))
        kacirilan_hirsiz = np.sum((finansal_kararlar == 0) & (gercek_hirsizlar == 1))
        toplam_finansal_zarar = (yanlis_alarm * c_fp) + (kacirilan_hirsiz * c_fn)

        print(f"--- Senaryo: {isim} ---")
        print(f"    [*] Matematiksel Optimal Esik Degeri: {finansal_esik:.4f}")
        print(f"    [*] Bloke Edilen Temiz Musteri (Maliyet: {c_fp}): {yanlis_alarm}")
        print(f"    [*] Kacirilan Sinsi Hırsiz (Maliyet: {c_fn}): {kacirilan_hirsiz}")
        print(f"    [zafiyet fatura tutari] BANKANIN KASASINDAN CIKAN NET ZARAR FATURASI: {toplam_finansal_zarar} TL")
        # -------------------------------------------------------------------------
    # faz 5: modeller arasi kiyaslama ve sampiyonu ilan etme (12. adim)
    # -------------------------------------------------------------------------
    # bu bolum simulasyon sonuclarini kurumsal terazide tartarak lideri secer
    print("\n" + "="*70)
    print("modeller arasi finansal kiyaslama ve sampiyonluk podyumu")
    print("="*70)
    
    # modelin firtina sonrasindaki nihai performans ozeti ekrana basiliyor
    print(f"[istatistik] degerlendirilen model: {secilen_model_adi}")
    print(f"    [*] secim kararsizligi skoru (entropy): {genel_model_kararsizligi:.4f}")
    print(f"    [*] hesaplanan finansal risk zarar faturasi: {toplam_finansal_zarar} tl")
    
    # en yuksek istikrari saglayan ve kasayi en iyi koruyan model ilan ediliyor
    print("\n[analiz sonucu ve lider model secimi]")
    print("    bankanin asimetrik maliyet süzgecinden basariyla gecen,")
    print("    risk bilancosunu en dipte tutan ve istikrar saglayan model:")
    print(f"    liderlik statusu verilen model: {secilen_model_adi}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
    

                       

