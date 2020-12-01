from keras.models import load_model
import numpy as np
from keras.preprocessing import image

pneumonia=load_model("pcnn5.h5")

picture=image.load_img(r"#address of image to test",target_size=(64,64))

z= image.img_to_array(picture)
prediction=np.expand_dims(z,axiz=0)
pred=pneumonia.predict_classes(prediction)

print(pred)
