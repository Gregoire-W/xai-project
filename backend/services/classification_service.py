from core.constants import DEBUG_PATH
import numpy as np
import tensorflow as tf

class ClassificationService:

    def predict_audio(self, model, class_names, img):
        print("Start predict audio", flush=True)
        
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predictions = tf.where(predictions < 0.5, 0, 1)
        
        predictions = predictions.numpy().flatten()
        
        prediction_labels = [class_names[int(pred)] for pred in predictions]
        print(f"predicted labels: {prediction_labels}", flush=True)

        return prediction_labels

    def predict_image(self, model, class_names, img):
        print("Start predict image", flush=True)

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        print(f"model predicted: {predictions}")
        predictions = predictions[0]
        predictions = [1 if pred > 0.3 else 0 for pred in predictions]
        
        prediction_labels = [class_names[idx] for idx, pred in enumerate(predictions) if pred == 1]
        print(f"predicted labels: {prediction_labels}", flush=True)

        return prediction_labels

classification_service = ClassificationService()