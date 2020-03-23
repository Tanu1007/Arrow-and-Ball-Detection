import cv2
import numpy
import tensorflow..python.keras.models import load_model
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.applications.vgg19 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array

model = load_model('Arrow2242.h5')          #load your model...make sure path of the model is correct
cap= cv2.VideoCapture(0)                    #change this acc to port on which usb cam is connected

#img = load_img("D:/Arrow Detection/train/left/download.jpg", target_size=(224, 224))

while True:

    _,frame = cap.read()
    img=frame.copy()
    frame = cv2.resize(frame,(224,224))
    

    image = img_to_array(frame)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    image = preprocess_input(image)
    yhat = model.predict(image)
    np.argmax(yhat,axis=1)
    
    print("check")        #1-left,2-none,3-right
    print(np.argmax(yhat) + 1)
    cv2.imshow("Frame",img)
    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()
