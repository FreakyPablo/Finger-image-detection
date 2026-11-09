import numpy as np 
import cv2 as cv

#Make model appears 
model_path = "" #Donner path de l'image à analyser
model = cv.imread(model_path)


#--Traitement d'image--  
#Noir et Blanc 
ycrcb = cv.cvtColor(model, cv.COLOR_BGR2YCrCb)
mask = cv.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
mask = cv.medianBlur(mask, 5)

#Contours des doigts 
contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = max(contours, key=lambda x: cv.contourArea(x))
cv.drawContours(model, [contours], -1, (255,255,0), 2)

#Enveloppe convexe
hull = cv.convexHull(contours)
cv.drawContours(model, [hull], -1, (0, 255, 255), 2)


# convexity defect sont les points les plus éloignées de chaque partie de l'enveloppe convexe et a l'exterieur de la forme de base. 
hull = cv.convexHull(contours, returnPoints=False)
defects = cv.convexityDefects(contours, hull)

# Utilisation trigonométrie afin de reconnaitre les doigts
# c = sqrt(a^2 + b^2 - 2ab*cos(y))
# donc y = cos^-1((a^2+b^2 - c^2)/2ab)
# Convexity defect + trigo, Si angle inférieur a 90° alors c'est un doigt
# Défaut de convexité renvoyé sous forme [point départ, point arrivé, point plus éloigné, distance]

cnt = 0

if points is not None:
    for i in range(points.shape[0]):  
        s,e,f,d = points[i].flatten() #Utiliser flatten permet d'éviter erreur ! suggeré par l'ia ! 

        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])

        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

        if (2*b*c) != 0:  #Afin d'eviter division par 0
            cos_angle = (b**2 + c**2 - a**2) / (2*b*c) # Trigonométire
            angle = np.arccos(np.clip(cos_angle, -1, 1)) # bloque les valeurs avec clip car arccos plante si valeur > 1 

            depth = d/256.0 # Afin de filtrer les faux défaut liés au bruit
            if angle <= np.pi / 2:  # si angle < 90 alors c'est un doigt
                cnt = cnt + 1
                cv.circle(model, far, 4, [0, 0, 255], -1)
    if cnt > 0:
        cnt = cnt+1
cv.putText(model, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)

cv.imshow("hull", model)
cv.waitKey(0)
cv.destroyAllWindows()

