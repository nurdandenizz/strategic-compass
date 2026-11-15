from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#(logistic regression): tamamen dogrusal denklemlere ve duz cizgi geometrisine dayanan 
# parametrik istatistik

#(decision tree): mantıksal sorgulara ve "evet/hayır" dallanmalarına dayanan tekil kural tabanlı

#(random forest): birden fazla modelin kolektif akılla oy kullandıgı gelismis topluluk

# yuzlerce modeli aynı anda 50 turluk siber fırtınalara sokmak bilgisayarın kaynaklarını ve islem zamanını
# gereksiz yere yorardı. bu 3 model, birbirinden tamamen farklı matematiksel karakterlere sahip oldugu icin
# aynı sabit veri uzerinde bakıs acısı cesitliligi ve kıyaslama zenginligi saglar


def evet_hayir_modeli_olustur(seed_degeri):
    # max_depth'i ezberlememek adına 5 yapıldı 
    # derinlik 4 olunca yetersiz, 6 yapınca 64 oda; gereksiz derin overfitting
    # guclu yonu: cok hızlıdır, kararları net ve anlasılırdır.
    # zayıf yonu: asırı kırılgandır, sans ruzgarına gore anında ezber yapar.
    # sarsıntı durumu: cok yuksek, her seed degistiginde (farklı karıstırmada) 
    # (high variance) - guvenilmezdir.
    
    model = DecisionTreeClassifier(max_depth=5, random_state=seed_degeri)
    return model

def gelismis_topluluk_modeli_olustur(seed_degeri):
    # 50 tane bagımsız yapay zeka modelini (evet-hayır modelini) temsil eder.
    # guclu yonu: asırı guvenilmezligi bitirir, tek bir agacın hatasını digerleri kapatır (majority voting)
    # zayıf yonu: bilgisayarın ram'ini ve islemcisini cok yorar, agırdır.
    # sarsıntı durumu: cok dusuk, icindeki 50 agacın ortak aklı sayesinde  
    # seed degistirse bile pratikte neredeyse hic sapıtmaz.
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=seed_degeri)
    return model

def duz_cizgi_modeli_olustur(seed_degeri):
    # kıdemli model
    # max_iter=1000 verdik ki o cizgiyi en mukemmel yere oturtmak icin 1000 kere denesin.
    # guclu yonu: sarsıntılardan hic etkilenmez, cok az kaynak harcar.
    # en buyuk zaafı: veriler karmasıksa (egri bugru ise) dumduz cizgi cekemez, hirsizi kacırır.
    # sarsıntı durumu: sıfıra yakındır (near-zero variance), matematiksel olarak cekilecek tek bir dogru cizgiye 
    # odaklandıgı icin seed sifresini ne kadar degistirirsen degistir beton gibi sabit kalır.
    
    model = LogisticRegression(random_state=seed_degeri, max_iter=1000) 
    # sarsıntı sifresi makineye baglandı ve tur sınırı 1000 olarak belirlendi
    return model
