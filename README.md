# Image_OCR_CRNN
Arabic ID first name OCR using CRNN with CTC loss
Create content folder and download the saved model weights from here: https://drive.google.com/file/d/1ED_j7NfWL22FpNGJcMAjFGKqVOiifxhc/view?usp=drive_link , or you can start training directly using training notebook.
You can also use django  web app to test the model inference directly, just copy the model weights and paste the file in static folder, then add the required image to the static folder, run Django and enter the name of the image you want to test.

Current model metrics values: loss: 0.3184 - accuracy: 0.8861 - val_loss: 1.6371 - val_accuracy: 0.7852

### data.py file
The file contains class ReadData which is used to read all of the text files in a specified folder and return a dictionary where the keys are the file names and the values are the file contents.

### image_utils.py file
Python class called ImageProcessingUtils. It provides a number of methods for processing images, including reading, preprocessing, cropping, and resizing.
The read_image method reads an image in the gray scale given path.
Then after the image is read it’s passed to preprocess_image method, it performs a number of operations on the image to make it easier to extract text, such as converting it to grayscale, applying a threshold, and removing noise.
After the preprocess_image method, the crop_to_text method crops the image to the region that contains text by using The findNonZero method which finds all the non-zero pixels in an image.
Finally to reshape the images into the same shape but at the same time maintaining the aspect ratio.

###  text_utils.py file
Contains three functions: encode_to_labels, labels2text, and pad_text. These functions are used to transform text into numbers, numbers into text, and pad text with a specific value.

### model_utils.py 
The provided code implements a Convolutional Recurrent Neural Network (CRNN) for text recognition. The model architecture combines convolutional layers for feature extraction with recurrent layers for sequence modeling.

### Django inference snapshot
![inference](https://github.com/mahmoudmohey97/Image_OCR_CRNN/assets/34627623/8b469936-7aa0-4fa9-882e-d1090cdf058f)

