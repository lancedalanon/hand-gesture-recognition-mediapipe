import os
import sys
import numpy as np
import tensorflow as tf

class PointHistoryClassifier(object):
    def __init__(
        self,
        model_path=None,  
        score_th=0.5,
        invalid_value=0,
        num_threads=1,
    ):
        # Determine base path dynamically depending on environment
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS  
        else:
            base_path = os.path.abspath(".") 

        # Use dynamic path if no custom model path is provided
        if model_path is None:
            model_path = os.path.join(base_path, 'model\\point_history_classifier\\point_history_classifier.tflite')

        # Load the TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)

        # Allocate tensors and get input/output details
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        point_history,
    ):
        # Prepare the input for the interpreter
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_tensor_index,
                                    np.array([point_history], dtype=np.float32))

        # Invoke the model
        self.interpreter.invoke()

        # Get the output and determine the result
        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Get the index of the result with the highest probability
        result_index = np.argmax(np.squeeze(result))

        # If the score is below the threshold, return the invalid value
        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index
