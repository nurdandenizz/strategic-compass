import argparse
import numpy as np
import pandas as pd
from src.models import bilge_agac_modeli_olustur, konsey_modeli_olustur, eski_toprak_modeli_olustur
# todo: bu satir bizim yan odadaki src/models.py dosyasiyla kurdugumuz gizli tunel.
# imalathanede urettigimiz 3 canavar makineyi bu ana santiye alanina (experiment_runner.py)
# baglayan dev uzatma kablosudur. bu kabloyu cekmezsek bilgisayar makineleri taniyamaz hata verir.

def sahte_banka_verisi_uret():
    # todo: module 10 icin zaman sirali sahte veri uretiyoruz.
    # elimizde 1000 tane banka islemi olsun.
    
    # 42 ile 34 farki: matematiksel kalitesi yok sadece karistirma sifresi. 34 de yazabilirdik.
    # ama yazilim dunyasinda otostopçunun galaksi rehberi kitabina ithafen "hayatin ve evrenin 
    # sirri" geyigi olarak herkes 42 yazar, bu bir sektor kulturudur.
    # bunu sabitledik ki bilgisayar her seferinde ayni sahte veriyi uretsin.
    # sonra module 5'te bu sifreyi 50 kez degistirip modele yapay deprem yaratacagiz.
    np.random.seed(42)
    
    # cuzdan simulatoru: 10 tl ile 10000 tl arasinda rastgele 1000 tane harcama parasi.
    # buradaki uniform kelimesi esit dagilimli demektir yani bilgisayar sayilari secerken 
    # hep kucuk paralar veya hep buyuk paralar uretmez. 10 tl ile 10000 tl arasindaki 
    # her sayinin secilme sansi esittir. boylece gercek hayattaki gibi hem sakiz alan adamin 
    # harcamasi hem de son model telefon alan adamin harcamasi ayni deftere yuklenir.
    tutar = np.random.uniform(10, 10000, 1000)
 
    # 0 gunduz, 1 gece demek. p=[0.8, 0.2] olasilik ayari.
    # islemlerin %80'i gunduz %20'si gece olsun dedik cunku gercek hayatta da boyledir.
    # sinsi hackerlar genelde o azinlikta kalan gece saatlerini secer.
    gece_mi = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
    
    # 0 temiz islem, 1 dolandirici (fraud) demek. p=[0.95, 0.05] ayari.
    # bilerek %95 temiz %5 hirsiz yaptik cunku bankada islemlerin yarisi hirsizlik olamaz.
    # sinsi fraud vakalari veri denizinde asiri nadirdir (imbalanced data).
    # modellerimiz samanlikta igne aramayi ogrensin diye bu gercekci orani sectik.
    hedef = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
    
    # dataframe ne demek: havada ucusan bu 1000'er tane sayiyi bizim muhasebeci (pandas)
    # yakaliyor ve excel tablosu gibi tertemiz bir hesap defterine diziyor.
    veri = pd.DataFrame({"tutar": tutar, "gece_mi": gece_mi, "hedef": hedef})
    return veri

def main():  # ana kontrol fonksiyonu, is akisi siralamasi buradan yonetilir
    parser = argparse.ArgumentParser(description="strategic compass - model selection benchmark")  # komut yakalayici nesne olusturuldu ve proje aciklamasi tanimlandi
    parser.add_argument(
        "--model",  # terminalden girilecek ana parametre ismi tanimlandi
        type=str,  # girilen parametre degerinin veri tipi metin olarak belirlendi
        required=True,  # bu parametrenin girilmesi zorunlu kilindi, girilmezse kod calismaz
        choices=["lr", "dt", "rf"],  # girilebilecek parametre degerleri bu uc secenekle sinirlandirildi
        help="secenekler: lr (logistic regression), dt (decision tree), rf (random forest)"  # yardim komutunda ekrana basilacak aciklama metni
    )
    
    args = parser.parse_args()  # kapidaki gorevli (parser) komutlari parcalarina ayirdi (parse) ve args kutusuna kaydetti
    print("[info] benchmark motoru basariyla baslatildi...")  # sistemin calismaya basladigini belirten bilgi mesaji
    print(f"[info] secilen model: {args.model.upper()}")  # secilen parametre degerini buyuk harfe cevirip ekrana basan satir
    
    # veri yukleme odasi tetikleniyor
    banka_verileri = sahte_banka_verisi_uret()
    print(f"[info] module 10: {len(banka_verileri)} tane zaman sirali islem yuklendi.")
    
    # =========================================================================
    # nd'nin zaman duvari (temporal validation split)
    # =========================================================================
    # bazilari veriyi karistirip gelecekteki taktikleri gecmise sizdirir.
    # biz araya kalin bir zaman duvari cektik. ilk 750 islemi ogrenmeye ayirdik,
    # kalan son 250 islemi ise gelecek simulasyonu (test) icin sakliyoruz.
    # amacimiz modelin gecmisi ezberleyen bir papagan mi yoksa gelecekteki yepyeni
    # anomali turlerini bile yakalayabilen bir akilli mi oldugunu olcmek.
    gecmis_egitim_verisi = banka_verileri.iloc[:750]
    gelecek_test_verisi = banka_verileri.iloc[750:]
    
    print(f"[info] zaman duvari cekildi! egitim verisi: {len(gecmis_egitim_verisi)}, test verisi: {len(gelecek_test_verisi)}")
    # =========================================================================
    
    if args.model == "lr":  # eger terminalden girilen parametre lr ise bu kosul blogu calisir
        model = eski_toprak_modeli_olustur(seed_degeri=42)  # yan odadan logistic regression model nesnesi cagirildi
        print("[info] eski toprak modeli basariyla tetiklendi, beton gibi hazir.")  # ekrana lr modelinin yuklendigini basan bilgi satiri
    elif args.model == "dt":  # eger terminalden girilen parametre dt ise bu kosul blogu calisir
        model = bilge_agac_modeli_olustur(seed_degeri=42)  # yan odadan decision tree model nesnesi cagirildi
        print("[info] tek beyinli bilge agac tetiklendi, sarsintiya acik bekliyor.")  # ekrana dt modelinin yuklendigini basan bilgi satiri
    elif args.model == "rf":  # eger terminalden girilen parametre rf ise bu kosul blogu calisir
        model = konsey_modeli_olustur(seed_degeri=42)  # yan odadan random forest model nesnesi cagirildi
        print("[info] 50 agaclik canavar konsey tetiklendi, oy coklugu mekanizmasi aktif.")  # ekrana rf modelinin yuklendigini basan bilgi satiri

if __name__ == "__main__":  # bu dosyanin terminalden dogrudan ana program olarak calistirilip calistirilmadigini denetleyen sart
    main()  # dosya terminalden dogrudan tetiklendiyse ana kontrol fonksiyonunu baslatan ana salter satiri
