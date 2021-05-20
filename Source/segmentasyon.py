import cv2
import numpy as np
from model import Deeplabv3

deeplab_model = Deeplabv3()
video = cv2.VideoCapture(0)


blurKernel = (41,41)
while True:
    basarili,kare = video.read()
    if basarili ==True:
        
        genislik, yukseklik, _ = kare.shape
        oran = 512. / np.max([genislik,yukseklik])
        kare_input = cv2.resize(kare,(int(oran*yukseklik),int(oran*genislik))) / 127.5 - 1.
        genislik = kare_input.shape[0]
        pad_genislik = int(512 - genislik)
        kare_input = np.pad(kare_input,((0,pad_genislik),(0,0),(0,0)),mode='constant')    
        tahmin = deeplab_model.predict(np.expand_dims(kare_input,0))
        katmanlar = np.argmax(tahmin.squeeze(),-1)    
        katmanlar = katmanlar[:-pad_genislik-25]
        mask = katmanlar == 0
        katman_genislik, katman_yukseklik = katmanlar.shape
        kare = cv2.resize(kare, (katman_yukseklik,katman_genislik))
        bulaniklik = cv2.GaussianBlur(kare,blurKernel,0)
        kare[mask] = bulaniklik[mask]
        cv2.imshow("Bilgisayarli Goru |Arkaplan Bulaniklastirma",kare)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
video.release()
cv2.destroyAllWindows()