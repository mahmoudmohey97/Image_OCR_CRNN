from ocr_web import img_utils, model_utils, text_utils
from ocr_web.settings import BASE_DIR
import numpy as np

# custom method for generating predictions
def getPredictions(img_path):
    try:
        # Read image and apply preprocessing
        image = img_utils.ImageProcessingUtils().processing_pipeline(str(BASE_DIR)+"\\static\\"+img_path)
        image = image[np.newaxis, :, :]

        # Create model and load weights
        inference_model = model_utils.create_model_architecture()[0]
        inference_model.load_weights(str(BASE_DIR)+"\\static\\best_model_v2.hdf5")

        # Predict
        prediction = inference_model.predict(image)

        # Decode output
        out = model_utils.ctc_decoder(prediction)[0]
        
        ocr_text = text_utils.labels2text(out)
        return ocr_text

    except Exception as e:
        return "Error " + str(e)