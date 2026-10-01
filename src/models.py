from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def bilge_agac_modeli_olustur(seed_degeri):
    # todo: buradaki max_depth'i bilerek 5 yaptik ezberlemesin diye.
    #  derinlik 4 olunca yetersiz kaliyor, 6 yapinca 64 odaya cikip veri azligindan tesadufi sayilari ezberliyor 5 derinlik 32 odayla en ideal denge noktasidir.
    # 6 yapinca oda sayisi 64'e cikiyor ve sistem gg (overfitting).
    # sektorde lkbi:  diktator
    # guclu yn: cok hizlidir, kararlari net ve anlasilirdir.
    # en buyuk zaafi: asiri kirilgandir, sans ruzgarina gore aninda ezber yapar.
    # sarsinti durumu: cok yuksek, her seed degistiginde (farkli karistirmada) 
    # tamamen baska kararlar verip sapitir (high variance). guvenilmezdir.
    
    model = DecisionTreeClassifier(max_depth=5, random_state=seed_degeri)
    return model

def konsey_modeli_olustur(seed_degeri):
    # todo: bu da 50 tane bilge agactan kurdugumuz konsey modeli.
    # yukaridaki tek agac her depremde (seed degistikce) hemen cizgisini kaydiriyor.
    #
    # sektordeki lakabi: dev juri heyeti
    # guclu yonu: asiri guvenilmezligi bitirir, tek bir agacin hatasini digerleri kapatir (majority voting).
    # en buyuk zaafi: bilgisayarin ram'ini ve islemcisini cok yorar, agirdir.
    # sarsinti durumu: cok dusuk, icindeki 50 agacin ortak akli sayesinde 
    # seed degitirse bile pratikte neredeyse hic sapitmaz.
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=seed_degeri)
    return model

def eski_toprak_modeli_olustur(seed_degeri):
    # todo: bu da bankacilarin en eski dostu olan duz cizgi modeli.
    # max_iter=1000 verdik ki o cizgiyi en mukemmel yere oturtmak icin 1000 kere denesin.
    #
    # xsektordeki lakabi: eski toprak mufettis
    # guclu yonu: sarsintilardan hic etkilenmez (zero variance), cok az kaynak harcar.
    # en buyuk zaafi: veriler karmasiksa (egri bugru ise) dumduz cizgi cekemez, hirsizi kacirir.
    # sarsinti durumu: sifir! matematiksel olarak cekilecek tek bir dogru cizgi oldugu icin 
    # seed sifresini ne kadar degistirirsen degistir beton gibi sabit kalir.
    
    model = LogisticRegression(random_state=seed_degeri, max_iter=1000) 
    # sarsinti sifresi makineye baglandi ve tur siniri 1000 olarak belirlendi
    return model 
# ic mimarisi ve ayarlari tamamlanmis olan model nesnesi fonksiyonun disina firlatildi