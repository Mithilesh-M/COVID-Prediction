import cv2
import os
import numpy as np
from keras.models import model_from_json
from GUI import Dialogue_Positive
from GUI import Dialogue_Negative
def predict(path,img):
    user_input=[]
    img_size = 256
    #path='C:/Users/msmit/Desktop/Internship/New Dataset_1/COVID'
    #img='0a7faa2a.jpg'
    #path="C:/Users/msmit/Desktop/Internship/New Dataset_1/Normal"
    #img="IM-0115-0001.jpeg"
    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
    user_input=np.array(resized_arr)
    user_input = user_input.reshape(-1, img_size, img_size, 1)

    # load json and create model
    json_file = open('model_X_Ray.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_X_Ray.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    predicted_result = loaded_model.predict_classes(user_input)
    predicted_result = predicted_result[0][0]
    if(predicted_result==1):
        Dialogue_Positive.run(0)
    else:
        Dialogue_Negative.run()

#predict("C:/Users/msmit/Desktop/Internship/New Dataset_1/COVID","0a7faa2a.jpg")
#predict("C:/Users/msmit/Desktop/Internship/New Dataset_1/Normal","IM-0115-0001.jpeg")