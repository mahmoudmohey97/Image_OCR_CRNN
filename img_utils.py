import cv2
import numpy as np

class ImageProcessingUtils:
    def __init__(self, crop_shape= (128,32)):
        self.target_image_size = crop_shape
    
    def read_image(self, image_path):
        #read into grayscale
        return cv2.imread(image_path, 0)
    
    def preprocess_image(self, image):
        filtered_image = cv2.adaptiveThreshold(image,
                            100, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, 13, 16)
        kernel = np.ones((3, 3), np.uint8)
        # Removing small objects from an image, as well as for cleaning up noise.
        open_filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel)
        # Filling in small holes in the foreground objects & remove small gaps between foreground objects.
        close_filtered_image = cv2.morphologyEx(open_filtered_image, cv2.MORPH_CLOSE, kernel)
        binary = 255*(close_filtered_image < 50).astype(np.uint8)
        return binary
    
    def crop_to_text(self, image):
        # Find all non-zero points (text)
        coords = cv2.findNonZero(image)
        
        # Find minimum spanning bounding box
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop the image
        y2 = y-10
        x2 = x-10
        if y2 < 0:
            y2 = y
        if x2 < 0:
            x2 = x
        
        cropped_image = image[y2:y+h+10, x2:x+w+10]
        return cropped_image

    # This function is to resize while keeping aspect ratio, taken from this link
    # https://stackoverflow.com/questions/44720580/resize-image-to-maintain-aspect-ratio-in-python-opencv
    def reshape_image(self, img, padColor=0):
        h, w = img.shape[:2]
        sw, sh = self.target_image_size
        new_h = None
        new_w = None
        # interpolation method
        if h > sh or w > sw: # shrinking image
            interp = cv2.INTER_AREA

        else: # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = float(w)/h 
        saspect = float(sw)/sh

        if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
            new_h = sh
            new_w = np.round(new_h * aspect).astype(int)
            pad_horz = float(sw - new_w) / 2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0

        elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
            new_w = sw
            new_h = np.round(float(new_w) / aspect).astype(int)
            pad_vert = float(sh - new_h) / 2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0

        # set pad color
        if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
            padColor = [padColor]*3

        # scale and pad
        if new_h is None or new_w is None:
            new_h = sh
            new_w = sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
            
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
        return scaled_img
    
    def processing_pipeline(self, image_path):
        image = self.read_image(image_path)
        binary = self.preprocess_image(image)
        cropped_image = self.crop_to_text(binary)
        resized_image = self.reshape_image(cropped_image)
        return resized_image