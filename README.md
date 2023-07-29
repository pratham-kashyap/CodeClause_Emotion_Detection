# Emotion Detection using Deep Learning

This project is an implementation of Emotion Detection using Deep Learning with the help of Keras and OpenCV. It consists of two main files:

## Emotion_Detection.ipynb

The Jupyter Notebook `Emotion_Detection.ipynb` contains the code for creating, training, and evaluating the emotion detection model. It follows these steps:

1. Imports necessary libraries, including NumPy, Matplotlib, and Keras modules.

2. Loads the pre-trained MobileNet model without its top layer for feature extraction.

3. Adds a custom dense layer for emotion classification and compiles the model.

4. Sets up data generators for training and validation images to feed the model with real-time augmented data.

5. Implements early stopping and model checkpointing callbacks to monitor the model's performance during training and save the best model.

6. Trains the model using the training data and visualizes the training progress.

7. Loads the best-fit model from the saved checkpoint.

8. Uses the model to predict emotions for a single image and displays the result.

## Detector.py

The Python script `Detector.py` is responsible for real-time emotion detection from the webcam feed. The steps include:

1. Imports necessary libraries, including OpenCV and Keras modules.

2. Loads the pre-trained emotion detection model.

3. Initializes the Haar Cascade Classifier for face detection.

4. Captures frames from the default webcam using OpenCV's `VideoCapture` class.

5. Detects faces in each frame using the Haar Cascade Classifier and overlays rectangles around the detected faces.

6. Extracts the region of interest (ROI) containing the face, preprocesses it, and feeds it to the emotion detection model.

7. Predicts the emotion label for each face and overlays the predicted emotion on the frame.

8. Displays the webcam feed with emotions detected in real-time.

## How to Use

1. Make sure you have the required libraries installed. If not, install them using the following command: `pip install numpy matplotlib keras opencv-python`

2. Download the pre-trained model weights file `best_model.h5` and place it in the same directory as `Detector.py` and `Emotion_Detection.ipynb`.

3. To run emotion detection in real-time using the webcam, execute the `Detector.py` script: `python Detector.py` This will open a window showing the webcam feed with detected emotions displayed on the faces.

4. To train or explore the emotion detection model further, open and execute the `Emotion_Detection.ipynb` notebook using Jupyter Notebook or any compatible environment.

Feel free to modify the code, experiment with different models, or use the emotion detection functionality in your projects!

