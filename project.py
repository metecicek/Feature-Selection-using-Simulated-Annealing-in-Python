# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 21:47:53 2020

@author: İbrahim Mete Çiçek
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn import tree
from sklearn.model_selection import cross_val_score

# Veri setini işleme hazırlama kısmı
data = pd.read_csv('veri.csv') 
print(data.head())
X = data.iloc[:,:-1]
y = data.iloc[:,-1:].values.flatten()
kromozom_uzunlugu = len(data.columns)-1    


# Bu işlem iterasyon sayısını etkiler
sıcaklık = math.exp(3)   # Başlangıç sıcaklığı
minimum_sıcaklık = math.exp(-8)    # Sıcaklığı durdur
alfa_degeri = 0.98    # Soğutma katsayısı

iterasyon_sayısı = 10    # En kötü çözümlerin üretilme sayısı
# Kötü bir çözümü benimseme olasılığını etkiler, daha kötü bir çözümü benimsemek daha kolay olur
k = 0.002

def uygunluk(delta_E,sıcaklık):     
    if delta_E<=0:   # ΔE <= 0, ifadesi doğrudan kabul edildi
        print('Sonuç: Doğrudan Çocuk sahibi olabilir')
        return True

    p=math.exp(-delta_E/(k*sıcaklık))    # Çocuk sahibi olma olasılığı
    r=random.random()
    if r<p:
        print(str(r)+"<"+str(p)+"，Sonuç: Çocuk sahibi olma ihtimali var") 
        return True
    else:
        print(str(r)+">="+str(p)+"，Sonuç: Çocuk sahibi olamaz"+"（"+str(iterasyon_sayısı)+"）")
        return False

# Feature selection kodlaması, simulated annealing buradan itibaren başlıyor
def gen_kodlaması(): # Gen Kodlaması
    while True:
        temp = []
        kromozom_degeri_1 = False   # Bu kromozomun değerinin 1 olmasının sorgulanması
        for j in range(kromozom_uzunlugu):
            rand = random.randint(0,1)
            if rand == 1:
                kromozom_degeri_1 = True
            temp.append(rand)
        if kromozom_degeri_1:   # Kromozomların tümü 0 olamaz
            return temp
        
def model_tutarlıgı(x):
    X_test = X

    kromozom_degeri_1 = False
    for j in range(kromozom_uzunlugu):
        if x[j] == 0:
            X_test =X_test.drop(columns = j)
        else:
            kromozom_degeri_1 = True
    X_test = X_test.values
        
    if kromozom_degeri_1:     
        gtf = tree.DecisionTreeClassifier() # Karar ağacı model yapılandırması
        model_tutarlıgı = cross_val_score(gtf, X_test, y, cv=5).mean()  # 5 kez çapraz doğrulama optimizasyonu
        

        return model_tutarlıgı   
    else:
        model_tutarlıgı = 0     # Tüm modellerin uygunluğu 0'dır
        return model_tutarlıgı

# Eski çözümden yeni çözüm üretilmesi
def yeni_kromozom(x):
    mutasyon_noktası = random.randint(0, kromozom_uzunlugu-1)  # Rastgele mutasyon noktaları seçimi
    if x[mutasyon_noktası] == 1:
        x[mutasyon_noktası] = 0
    else:
        x[mutasyon_noktası] = 1          
    return x

# Programın görselleştirilmesinin giriş kısmı
if __name__=='__main__':
    plt.xlabel('sıcaklık')
    plt.ylabel('model_tutarlıgı')
    plt.xlim((minimum_sıcaklık,sıcaklık))    # X koordinat aralığı
    plt.ylim((0.4,0.9)) # Y koordinat aralığı
    px = []
    eski_py = []
    yeni_py = []
    plt.ion()
    
    eski_g = gen_kodlaması()   # İlk önce rastgele çözüm üretilmeli
    eski_m = model_tutarlıgı(eski_g)

    while sıcaklık > minimum_sıcaklık:

        yeni_g = yeni_kromozom(eski_g)   # Rastgele çözümlerin üretilmesi
        yeni_m = model_tutarlıgı(yeni_g)   # Model tutarlığı yani fitness fonksiyonu, fit etmek oturtmak anlamına geliyor, modelin tutarlığını gösteriyor
        delta_E = -(yeni_m - eski_m)

        if uygunluk(delta_E,sıcaklık):  # Kabul edilebilir
            eski_g = yeni_g
            eski_m = yeni_m 

        if delta_E<=0:   # ΔE<=0, soğutma işlemi
            sıcaklık = sıcaklık * alfa_degeri
        else:
            iterasyon_sayısı -= 1

        if iterasyon_sayısı < 0:
            break

        print(sıcaklık)
        print(eski_g)
        print(eski_m)
        print()

        px.append(sıcaklık)  # Elde edilen verilerin görsel olarak çizilme kısmı
        eski_py.append(eski_m)
        yeni_py.append(yeni_m)
        plt.plot(px,eski_py,'r')
        plt.plot(px,yeni_py)
        plt.show()
        plt.pause(0.001)
    
