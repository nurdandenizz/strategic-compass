import argparse
import numpy as np
import pandas as pd
from src.models import bilge_agac_modeli_olustur, konsey_modeli_olustur, eski_toprak_modeli_olustur
# todo: bu satir bizim yan odadaki src/models.py dosyasiyla kurdugumuz tnl.
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
    # amac modelin gecmisi ezberleyen bir papagan mi yoksa gelecekteki yeni
    # anomali turlerini bile yakalayabilen bir akil mi oldugunu olcmek.
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








    # -------------------------------------------------------------------------
    # FAZ 2: 50 TURLUK DEPREM DÖNGÜSÜ VE MODEL EĞİTİM MOTORU
    # -------------------------------------------------------------------------
    
    # todo: modelin 50 farkli firtinadaki tahmin olasiliklarini toplamak icin dev bir hafiza havuzu aciyoruz
    tum_turlar_tahmin_olasiliklari = [] # her turun 250 adetlik tahmini bu sepetin icine alt alta dizilecek

    # 50 turluk buyuk kusatma basliyor; her turda seed sifresi 1'den 50'ye kadar degisecek
        # sarsinti_seed: modelleri tek bir sansli gunde degil, 50 farkli yapay deprem altinda test etmek icin donen firtina mekanizmasidir.
    for sarsinti_seed in range(1, 51):
        
        # 1. adim: o turdaki sarsinti sifresine (seed) gore ana makineyi fabrikadan cagiriyoruz
        if secilen_model_adi == "LR":
            from src.models import eski_toprak_modeli_olustur
            aktif_model = eski_toprak_modeli_olustur(seed_degeri=sarsinti_seed)
        elif secilen_model_adi == "DT":
            from src.models import bilge_agac_modeli_olustur
            aktif_model = bilge_agac_modeli_olustur(seed_degeri=sarsinti_seed)
        elif secilen_model_adi == "RF":
            from src.models import canavar_konseyi_olustur
            aktif_model = canavar_konseyi_olustur(seed_degeri=sarsinti_seed)

        # 2. adim: o turda sarsilan gecmisteki 750 islemi makineye tek tek ezberletiyoruz
        # fit mekanizmasi: bilgisayar X_train tablosundaki harcama ve saat verilerini, y_train cevap anahtarindaki+
        #  0 ve 1 etiketleriyle ayni satir numaralari (indeks) uzerinden milimetrik olarak eslestirir ve raporlar.

        aktif_model.fit(X_train, y_train)

        # 3. adim: makineyi hic gormedigi son 250 test isleminin ustune salip hırsızlık olasiliklarini (1. sutun: fraud) cekiyoruz
        # [:, 1] yazarak sadece hirisizlik olasiligini havada yakaliyoruz
        # 
        # 
        # [:, 1]: modelin arka planda otomatik urettigi ikiz paketten (sol taraf: temizlik yuzdesi, sag taraf: hirsizlik yuzdesi)
        #  sol tarafi filtreleyip atar, sadece sagdaki sinsi hirsizlik yuzdelerini ceker.

        o_turun_tahminleri = aktif_model.predict_proba(X_test)[:, 1]

        # 4. adim: yakaladigimiz bu 250 adetlik sinsi olasilik skorunu buyuk hafiza havuzuna firlatiyoruz
        tum_turlar_tahmin_olasiliklari.append(o_turun_tahminleri)

    # todo: 50 tur bittiğinde elimizde her biri 250 tahminden olusan devasa bir veri matrisi birikmis olacak
    print(f"[!] {secilen_model_adi} modeli icin 50 turluk sarsinti ve gelecek simulasyonu basariyla tamamlandi!")





    # -------------------------------------------------------------------------
    # faz 3: selection entropy (secim kararsızlıgı) hesaplama motoru
    # -------------------------------------------------------------------------
    # 1. adim: elimizdeki o 50 satirlik lojistik havuzu matematiksel bir numpy matrisine donusturuyoruz
    # bu matrisin boyutu: 50 satir (depremler) x 250 sutun (gelecek test islemleri) olacak
    matris_tahminler = np.array(tum_turlar_tahmin_olasiliklari)

    # 2. adim: 250 test isleminin her biri icin 50 tur boyunca uretilen hirsizlik olasiliklarinin ortalamasini aliyoruz
    # axis=0 yazarak dikey eksende (50 deprem boyunca) ortalama hesapliyoruz
    # axis=0 pusulasi: bilgisayara musterilerin verilerini birbirine karistirmadan, her bir musterinin  
    # 50 farkli firtinadak skorlarini yukaridan asagiya dikey koridorlar halinde tarayip ortalamasını almasını soyler
    ortalama_p = np.mean(matris_tahminler, axis=0)

    # logaritma fonksiyonu sifir (0) gordugunde patlamasin diye ufak bir koruma ekliyoruz
    # 1. bir sonraki satırda o meşhur shannon entropy (log2) çarkı dönecektir.
    # 2. matematik kurallarına göre tam 0 (sıfır) sayısının logaritması tanımsızdır.
    # 3. eğer model 50 tur boyunca bir işlemden yüzde yüz emin olursa ortalaması tam 0.0 veya 1.0 çıkar.
    # 4. tam 0 veya 1 değerleri logaritma çarkına girerse python kilitlenir ve tüm raporlama çöker.
    # 5. np.clip komutu bu aşırı uç sayıları tırnak makası gibi milyonda bir hassasiyetle kırpar.
    # 6. tam 0'ları 1e-15 (0.000000000000001) seviyesine çeker, tam 1'leri ise 0.999999999999999 seviyesine indirir.
    # 7. böylece matematik kuralları çiğnenmeden, bilgisayar patlamadan istatistik motoru güvenle döner.
    ortalama_p = np.clip(ortalama_p, 1e-15, 1 - 1e-15)

    # 3. adim: meshur shannon entropy formülünü matrisin uzerine saliyoruz
    # neden?
    # 1. modellerin 50 farklı fırtınadaki zikzaklarını tek bir net nota (0 ile 1 arasına) bağlamamız şarttır.
    # 2. sol taraf [ p * log2(p) ]: modelin o işlem için ürettiği sinsi 'hırsızlık' tereddüdünü ölçer.
    # 3. sağ taraf [ (1-p) * log2(1-p) ]: modelin o işlem için ürettiği otomatik 'temizlik' tereddüdünü ölçer.
    # 4. bu iki parça toplandığında, her işlem için 0 ile 1 arasında kurumsal bir kararsızlık puanı çıkar:
    #    - senaryo a: model 50 tur boyunca hep aynı karardaysa (p=0 veya p=1), terazi bunu görür ve 0.0 (sıfır kararsızlık) notu verir.
    #    - senaryo b: model tam bir kumarbaz gibi sürekli çark ettiyse (p=0.50), terazi zirve yapar ve 1.0 (maksimum kararsızlık) notu verir.
    # 5. kısacası bu formül, modelin şansa mı yoksa gerçek mantığa göre mi tahmin ürettiğini mühürleyen tek tarafsız dedektördür
    # formul: - [ p * log2(p) + (1-p) * log2(1-p) ]
    secim_entropisi = - (ortalama_p * np.log2(ortalama_p) + (1 - ortalama_p) * np.log2(1 - ortalama_p))

    # 4. adim: butun test seti uzerindeki o genel kararsizlik ortalamasini tek bir rapor skoruna indirgiyoruz
    genel_model_kararsizligi = np.mean(secim_entropisi)

    print(f"[+] {secilen_model_adi} Modeli Icin Genel Secim Kararsizligi (Entropy): {genel_model_kararsizligi:.4f}")




        # -------------------------------------------------------------------------
    # faz 4: cost-sensitive evaluation ( finansal süzgeci)
    # -------------------------------------------------------------------------
    # 1. adim: bankanin asimetrik acidan gercek dunya maliyetlerini koda tanimliyoruz
    # c_fn: hirsizi kacirmanin devasa cezasi (false negative maliyeti)
    # c_fp: temiz musteriyi yanlislikla bloke etmenin kucuk cezasi (false positive maliyeti)
    maliyet_senaryolari = [
        {"isim": "1:10 hafif kriz", "c_fn": 10, "c_fp": 1},
        {"isim": "1:20 agir kriz", "c_fn": 20, "c_fp": 1}
    ]

    print(f"\n[!] {secilen_model_adi} Icin Finansal Risk Raporu Inceleniyor...")

    # -------------------------------------------------------------------------
    # fınansal muhasebe v lojıstık ayıklama 
    # -------------------------------------------------------------------------
    # 1. for senaryo dongusu: yukarıda hazırladıgımız 2 farklı krız klasorunu sırayla acar.
    # 2. senaryo["isim"]: o an eldekı aktif klasorun uzerındekı ısım etıketıne bakar, metnı ceker.
    # 3. senaryo["c_fn"]: klasorun ıcındekı hirsizi kacirma malıyet faturasını (10 veya 20) ayıklar.
    # 4. senaryo["c_fp"]: klasorun ıcındekı musterıyı bosa engelleme taban bedelını (1) yakalar.
    # 5. bu satırlar, ayıklanan bu sayıları bir sonrakı adımdaki matematıksel terazıye besler.
    # -------------------------------------------------------------------------
    for senaryo in maliyet_senaryolari:

        # -------------------------------------------------------------------------
        # lojıstık ayıklama ve değışken kutuları gorev farkları (unpackıng)
        # -------------------------------------------------------------------------
        # neden bu satırlar var ve farkları ne?
        # 1. maliyet_senaryolari dev bir sepet listesidir; bu satırlar ise o sepetten cıkan parcalardır.
        # 2. isim kutusu (metin / raporlama): icine sadece kriz adını kaydeder, matematiksel hesaba girmez, sadece ekrana basılır.
        # 3. c_fn kutusu (ağır finansal ceza): hırsızı kacırmanın bankaya kestıgı agir para cezası tamsayısıdır (10 veya 20).
        # 4. c_fp kutusu (küçük operasyon bedeli): masumu bosa engellemenın getırdıgı 1 birimlik taban prestij kaybı bedelıdır.
        # 5. farkları: biri sadece raporlama yazısıyken, dıger ıkısı esık formulu ve zarar faturası hesaplayan fiziki sayılardır!
        # -------------------------------------------------------------------------
        isim = senaryo["isim"]
        c_fn = senaryo["c_fn"]
        c_fp = senaryo["c_fp"]

        # -------------------------------------------------------------------------
        # teoretık optimal esık terazısı felsefesı (mınımum cost equatıon)
        # -------------------------------------------------------------------------
        # neden bu formül?
        # 1. bankanın kasasından çıkacak toplam zarar faturasını matematiksel olarak en dip seviyeye indirmek için.
        # 2. formül, 'toplam risk havuzu içinde masumu üzmenin payı ne kadardır?' sorusunu tartar.
        # 3. c_fn (hırsızı kaçırma maliyeti) arttıkça, kesrin alt tarafı büyür ve çıkan eşik sonucu sıfıra yaklaşır.
        # 4. baraj aşağı büküldükçe model çok daha tetikte bekleyen 'korkak' bir güvenlik görevlisine dönüşür.
        #    - 1:10 senaryosunda: 1 / (1 + 10) = %9 şüphe barajı (model %9 bile hırsızlık kokusu alsa kartı kilitler).
        #    - 1:20 senaryosunda: 1 / (1 + 20) = %4.7 şüphe barajı (ağır kriz anında en ufak tereddütte bile acımaz).
        # 5. bu terazi eski %50 kuralını . 
        # -------------------------------------------------------------------------
        finansal_esik = c_fp / (c_fp + c_fn)

        # -------------------------------------------------------------------------
        # fınansal esıgı gecme sorgusu ve astype(ınt) muhurleme mekanızması
        # -------------------------------------------------------------------------
        # fınansal esıgı gectıyse ne demek?
        # 1. ıslemın 50 turdakı ortalama hirsizlik suphesının, bankanın koruma barajını asması demektır.
        #    - ornek: 1:10 senaryosunda baraj %9 iken bir ıslemın suphesi %15 cıkarsa, bu ıslem esıgı gecer.
        # 2. parantez ıcı (ortalama_p >= finansal_esik): barajı gecenlere true, altında kalanlara false yazar.
        # 3. .astype(int) muhru: true yazan tehlıkelı ıslemlerı zınk dıye '1' (bloke et / kartı kılıtle) koduna cevırır.
        # 4. barajın altında kalıp esıgı gecemeyen false islemlerı ıse '0' (temız / gecıs ıznı ver) koduna muhurler.
        # 5. boylece sısleme, soyut kelimeler yerıne banka altyapısının anlayacagı net sayısal emirler verılmıs olur
        # -------------------------------------------------------------------------
        finansal_kararlar = (ortalama_p >= finansal_esik).astype(int)

        # -------------------------------------------------------------------------
        # gercek cevap anahtari ve astype kalip degistirme fabrikasi
        # -------------------------------------------------------------------------
        # astype ne demek ve neden bu satiri ekledik?
        # 1. y_test: o son 250 islemin gercek hayatta hirsizlik mi temiz mi oldugunu bilen asil gizli cevap anahtaridir.
        # 2. astype: bilgisayarin hafizasindaki mantik kelimelerini (true/false) saf birer sayi kalibina dökme emridir.
        # 3. bilgisayar bu pres makinesini gorunce gercek hirsizliklari zınk diye '1', temiz musterileri '0' yapar.
        # 4. neden yapiyoruz?: bir alt satirdaki toplama motorunun (np.sum) hata sayilarini fırtına gibi toplayabilmesi icin.
        # 5. boylece modelin tahminleriyle bu resmi tamsayili kılavuzu tartıp bankanin net zarar faturasini kesebiliriz
        # -------------------------------------------------------------------------
        gercek_hirsizlar = y_test.astype(int)

        # -------------------------------------------------------------------------
        # kurumsal zafiyet ve  dedektörleri (financial audit engine
        # -------------------------------------------------------------------------
        # 1. yanlis_alarm satiri: 
        #    - (finansal_kararlar == 1): modelin korkup 'hırsız' damgası bastığı işlemlerdir.
        #    - (gercek_hirsizlar == 0): cevap anahtarında aslında 'masum' olan temiz müşterilerdir.
        #    - ortadaki & (ve) işareti: modelin panik yapıp haksız yere bloke ettiği masumları ayıklar.
        #    - np.sum: bu iftira dosyalarını yukarıdan aşağıya dikey koridorda toplar, adetini bulur.
        #    - finansal etkisi: bu sayı bir alt satırda c_fp (1 tl) taban prestij cezasıyla çarpılacaktır.
        # -------------------------------------------------------------------------
        yanlis_alarm = np.sum((finansal_kararlar == 1) & (gercek_hirsizlar == 0))

        # -------------------------------------------------------------------------
        # 2. kacirilan_hirsiz satiri:
        #    - (finansal_kararlar == 0): modelin uyku sersemi 'temiz / geçsin' dediği işlemlerdir.
        #    - (gercek_hirsizlar == 1): cevap anahtarında aslında sinsi birer 'hırsız' olan dolandırıcılardır.
        #    - ortadaki & (ve) işareti: modelin gözünden kaçıp bankayı soyan o sinsi delikleri ayıklar.
        #    - np.sum: bu kaçak hırsızlık dosyalarını dikey koridorda tek tek toplar, adetini bulur.
        #    - finansal etkisi: bankayı asıl batıran deliktir, c_fn (10 tl veya 20 tl) .
        # -------------------------------------------------------------------------
        kacirilan_hirsiz = np.sum((finansal_kararlar == 0) & (gercek_hirsizlar == 1))

        # 6. adim: bankanin kasasindan cikan net tl zarar faturasini kesiyoruz
        toplam_finansal_zarar = (yanlis_alarm * c_fp) + (kacirilan_hirsiz * c_fn)

        print(f"--- Senaryo: {isim} ---")
        print(f"    [*] Matematiksel Optimal Esik Degeri: {finansal_esik:.4f}")
        print(f"    [*] Bloke Edilen Temiz Musteri (Maliyet: {c_fp}): {yanlis_alarm}")
        print(f"    [*] Kacirilan Sinsi Hırsiz (Maliyet: {c_fn}): {kacirilan_hirsiz}")
        print(f"    [zafiyet fatura tutari] BANKANIN KASASINDAN CIKAN NET ZARAR FATURASI: {toplam_finansal_zarar} TL")
