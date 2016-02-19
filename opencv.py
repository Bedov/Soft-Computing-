import numpy as np
import cv2
import sys, os

dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'resursi')
#treningpath = os.path.join(dir, 'trening')
treningpath = dir + '//trening'
testpath = dir + '//Test'

cascPath = os.path.join(dir, 'resursi/haarcascade_frontalface_default.xml')

faceCascade = cv2.CascadeClassifier(cascPath)

recognizer = cv2.createLBPHFaceRecognizer()

images = []
labels = []

image_paths = [os.path.join(treningpath, f) for f in os.listdir(treningpath)]

for image_path in image_paths:
    image_read = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
    image = np.array(gray_image, 'uint8')
    nbr = int(os.path.split(image_path)[1].split(".")[0])
    
    faces = faceCascade.detectMultiScale(
    image,
    scaleFactor=1.1,
    minNeighbors=20,
    minSize=(80, 80),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    
    #print "Nadjenih lica: {0} !".format(len(faces))

    for (x, y, w, h) in faces:
        images.append(image[y: y + h, x: x + w])
        labels.append(nbr)
        print "Dodajem: {0}. osobu!".format(nbr)
        cv2.imshow("Dodavanje lica za trening...", image[y: y + h, x: x + w])
        cv2.waitKey(50)
cv2.destroyAllWindows()

#Treniranje 
recognizer.train(images, np.array(labels))

#Fotografije za trening
image_paths_test = [os.path.join(testpath, f) for f in os.listdir(testpath)]

for image_path_test in image_paths_test:
    predict_image_read = cv2.imread(image_path_test)
    predict_gray_read = cv2.cvtColor(predict_image_read, cv2.COLOR_BGR2GRAY)
    
    predict_image = np.array(predict_gray_read, 'uint8')
    
    faces = faceCascade.detectMultiScale(
    image,
    scaleFactor=1.1,
    minNeighbors=8,
    minSize=(80, 80),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        
        nbr_actual = int(os.path.split(image_path_test)[1].split(".")[0])
        print nbr_actual

        if nbr_actual == nbr_predicted:
            print "osoba {}. je prepoznata sa ubedjenoscsu {}".format(nbr_actual, conf)
            cv2.rectangle(predict_gray_read, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if nbr_actual == 1:
                cv2.putText(predict_gray_read,"ID = Nenad!", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            if nbr_actual == 2:
                cv2.putText(predict_gray_read,"ID = Sale!", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        else:
            print "{} je pogresno prepoznata kao {}".format(nbr_actual, nbr_predicted)

        cv2.imshow("Prepoznavanje osobe...", predict_gray_read)
        cv2.waitKey(0)

        
        cv2.destroyAllWindows()






    
