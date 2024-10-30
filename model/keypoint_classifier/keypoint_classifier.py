import os
import sys
import numpy as np
import tensorflow as tf

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path=None, 
        num_threads=1,
    ):
        # Determine base path depending on environment
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")

        # Use dynamic path if no custom model path is provided
        if model_path is None:
            model_path = os.path.join(base_path, 'model\\keypoint_classifier\\keypoint_classifier.tflite')

        # Load the TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)

        # Allocate tensors and get input/output details
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        # Prepare the input for the interpreter
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_tensor_index,
                                    np.array([landmark_list], dtype=np.float32))

        # Invoke the model
        self.interpreter.invoke()

        # Get the output and determine the result
        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        return result_index