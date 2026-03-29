from keras.models import load_model

model = load_model("keras_Model.h5", compile=False)
print("Model loaded OK")