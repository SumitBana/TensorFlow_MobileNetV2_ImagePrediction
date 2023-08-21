import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

model = MobileNetV2(weights='imagenet')
def recognize_image(imgpath):
    img = image.load_img(imgpath, target_size=(224, 224))
    imgarr = image.img_to_array(img)
    imgarr = preprocess_input(imgarr)
    imgarr = tf.expand_dims(imgarr, axis=0)
    prediction = model.predict(imgarr)
    pred = decode_predictions(prediction)
    toppred = pred[0][0]
    conf = toppred[2]*100
    print("\nPredicted object:", toppred[1], "\nConfidence: {:.2f}% \n\n".format(conf))

imgpath= input("Enter the path to the image: ")
recognize_image(imgpath)
