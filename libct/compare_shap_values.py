
from __future__ import annotations
import sys
sys.path.insert(0, '/mnt/c/Users/user/Desktop/pyct_shap_value/PyCT-shapValue')
sys.path.insert(1, '/mnt/c/Users/user/Desktop/pyct_shap_value/PyCT-shapValue/libct')

import functools
import shap
from keras import Model
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import InputLayer
from typing import Tuple
import numpy as np

from libct.constraint import Constraint

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable
    class ShapValuedConstraint(Tuple[Constraint, float]):
        pass

class ShapValuesCalculator:
    def __init__(self, model: Model, background_dataset, input) -> None:
        self.model = model
        self.background_dataset = background_dataset
        self.input = input
        self.shap_values: dict[str, float] = dict()
        self.calculate_shap_values_for_all_neurons()

    def calculate_shap_values_for_all_neurons(self) -> None:
        trimmed_model = self.model
        transformed_background = self.background_dataset
        transformed_input = self.input
        number_of_layers = len(self.model.layers)
        for layer_number in range(0, number_of_layers):
            self.calculate_shap_values(
                trimmed_model, transformed_background, transformed_input, layer_number)
            transformed_input = ShapValuesCalculator.apply_one_layer( trimmed_model, transformed_input)
            transformed_background = ShapValuesCalculator.apply_one_layer_to_dataset( trimmed_model, transformed_background)
            if layer_number == number_of_layers - 1: break
            trimmed_model = ShapValuesCalculator.without_first_layer(trimmed_model)
    def recalculate_shap_values_for_all_neurons(self, new_input) -> None:
        self.input = new_input
        self.recalculate_shap_values_for_all_neurons()

    @staticmethod
    def without_first_layer(original_model: Sequential | Model) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            new_model = Sequential()
            for layer in original_model.layers[1:]:
                new_model.add(layer)
            new_model.build(original_model.layers[1].input_shape)
        elif isinstance(original_model, Model):
            # Get the output of the second layer
            new_input = original_model.layers[1].output
            # Get the output of the last layer
            new_output = original_model.layers[-1].output
            new_model = Model(inputs=new_input, outputs=new_output)
        return new_model

    @staticmethod
    def apply_one_layer_to_dataset( original_model: Sequential | Model, dataset: np.ndarray) -> np.ndarray:
        model_with_only_first_layer = ShapValuesCalculator.get_model_with_only_first_layer(
            original_model)
        model_with_only_first_layer.build(original_model.layers[0].input_shape)
        new_dataset = model_with_only_first_layer.predict(dataset)
        return new_dataset
    
    @staticmethod
    def apply_one_layer( original_model: Sequential | Model, original_input: np.ndarray) -> np.ndarray:
        return ShapValuesCalculator.get_model_with_only_first_layer(original_model).predict(original_input)

    @staticmethod
    def get_model_with_only_first_layer(original_model: Sequential | Model) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            model_with_only_first_layer = Sequential(original_model.layers[:1])
        elif isinstance(original_model, Model):
            model_with_only_first_layer = Model(
                inputs=original_model.input, outputs=original_model.layers[0].output)
        return model_with_only_first_layer
    
    def calculate_shap_values(self, model, background_dataset, input, layer_number: int) -> None:
        explainer = shap.DeepExplainer(model, background_dataset)
        shap_values = explainer.shap_values(input)
        for indices, shap_value in np.ndenumerate(shap_values):
            indices = indices[2:] # remove the first two dimensions
            self.shap_values[ShapValuesCalculator.get_position_key(
                layer_number - 1, indices)] = shap_value

    def get_shap_value(self, layer_number: int, indices: tuple[int, ...]) -> float:
        number_of_layers = len(self.model.layers)
        if layer_number == number_of_layers - 1:
            return float('inf')
        return self.shap_values[ShapValuesCalculator.get_position_key(layer_number, indices)]

    @staticmethod
    def get_position_key(layer_number: int, indices: tuple[int, ...]) -> str:
        key = str(layer_number)
        for i in indices:
            key += "_" + str(i)
        return key
    def print_shap_values(self) -> None:
      for key in self.shap_values:
        print(key, self.shap_values[key])

    

# usage:
# positioned_constraints: List[PositionedConstraint] = ...
# constraint: Constraint = pop_the_most_important_constraint(positioned_constraints, ShapValuesComparator().compare)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_shape = (28, 28, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 將標籤進行 one-hot 編碼
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

class ShapValuesCalculatorTester:
    def __init__(self):
        # construct a model with three Layers
        self.input_data = np.random.rand(1000, 10)
        self.model = Sequential()
        self.model.add(Dense(units=32, activation='relu', input_shape=(10,)))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
        self.labels = np.random.randint(2, size=(1000, 1))
        self.model.fit(self.input_data, self.labels, epochs=10, batch_size=32)
        self.background_dataset = self.input_data[np.random.choice(self.input_data.shape[0], 100, replace=False)]
        #self.background_dataset = self.background_dataset.reshape((100, -1))
        # create an input of floats between 0 and 1
        self.input = self.input_data[:1]
        # create a comparator
        self.comparator = ShapValuesCalculator(
            self.model, self.background_dataset, self.input)
        print(self.comparator.shap_values)

if __name__ == "__main__":
    ShapValuesCalculatorTester()