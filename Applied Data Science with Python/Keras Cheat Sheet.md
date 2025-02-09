# References for learning Keras

- [_Hands-on Machine Learning_](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/): (Géron) and companion [repository](https://github.com/ageron/handson-ml3)
- [_TensorFlow Tutorials_](https://www.tensorflow.org/tutorials)**:** Official tutorials covering various aspects of TensorFlow, from basics to advanced techniques
- [_Keras Documentation_](https://keras.io/)**:** Comprehensive guides and tutorials for building neural networks with Keras

## `Sequential()` Neural Networks

### Sequential Model

The Sequential model in Keras is akin to a straight line of layers, perfect for neural networks that follow a direct layer-after-layer structure. Initiate by creating a Sequential model instance, then layer it with various types such as dense (fully connected), convolutional, and dropout layers tailored to your needs.

```Python
from keras.models import Sequential
model = Sequential()
```

### Adding Layers

To construct the network, layers are stacked in the Sequential model, with the inaugural layer needing input shape details.

```Python
from keras.layers import Dense, Conv2D, Flatten

# Incorporating a dense layer
model.add(Dense(units=64, activation='relu', input_dim=100))

# Introducing a convolutional layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

# Flattening for dense layers
model.add(Flatten())
```

### Compilation

Pre-training configuration is set through the `compile` method, specifying the optimizer, loss function, and metrics.

**Optimizers** adjust the neural network attributes, such as weights and learning rate, to minimize losses.

- **Adam:** Known for its adaptive learning rate capabilities, making it suitable for a wide range of problems.
- **SGD (Stochastic Gradient Descent):** A traditional optimizer that is simple yet effective, especially with a well-tuned learning rate.

**Activation functions** determine a node's output based on its inputs, crucial for introducing non-linearity into the model to learn complex patterns.

- **ReLU (Rectified Linear Unit):** Commonly used for its efficiency and effectiveness.
- **Softmax:** Ideal for the output layer in multi-class classification, converting logits into probabilities.
- **Sigmoid:** Often used in binary classification, transforming values to a range between 0 and 1.

```Python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Training

Models train on Numpy arrays of input data and labels with the `fit` method.

```Python
# Given x_train and y_train as your data and labels
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

### Evaluation

Model performance is gauged on a test dataset via the `evaluate` method.

```Python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

### When to Use Sequential Models:

- **Use Case**: Ideal for scenarios with a singular input source and output destination, and when the model's structure is a linear layer sequence.
- **Limitations**: Inadequate for networks needing shared layers, multiple inputs/outputs, or branching layers. For such complexities, the Functional API is recommended.

## Non-Sequential and Advanced Architectures

- The Functional API, offering flexibility for intricate architectures, supports non-linear topologies, shared layers, and multi-input/output models. While potent, its complexity may be better suited for seasoned users or specific cases. For many standard tasks and beginners, the Sequential model provides a simpler yet effective approach to building neural networks.

## Common Types of Layers

In Keras, layers are the building blocks of neural networks, and choosing the right types of layers is crucial for your model's performance. Here are some common types of layers used in Keras:

**Dense Layers:**

- Fully connected layers where each neuron in the layer is connected to all neurons in the previous layer.

```Python
Dense(units, activation=None)
```

- `units`: Number of neurons in the layer.
- `activation`: Activation function to use. Common choices include 'relu', 'sigmoid', and 'softmax'.

**Convolutional Layers (Conv2D):**

- Used primarily in image processing for feature extraction.

```Python
Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', activation=None)
```

- `filters`: Number of output filters in the convolution.
- `kernel_size`: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
- `strides`: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width.
- `padding`: One of 'valid' or 'same' (case-insensitive).
- `activation`: Activation function to use.

**Flatten Layers:**

In neural network models, especially those dealing with image data, it is common to use convolutional layers (`**Conv2D**`) to process the image input. However, when transitioning from convolutional layers to fully connected layers (`**Dense**`), there is a need to flatten the multidimensional output of the convolutional layers into a single dimension. This is where Flatten layers come into play.

```Python
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(**Flatten()**)  # Flattens the output of the Conv2D layer
model.add(Dense(256, activation='relu'))
```

In this example, the `**Flatten**` layer takes the output of the `**Conv2D**` layer, which is a 3D tensor (excluding the batch dimension), and flattens it into a 1D tensor. This flattened output then serves as the input to the subsequent `**Dense**` layer.

**Pooling Layers (MaxPooling2D):**

- Used to reduce the spatial dimensions (height and width) of the input volume.

```Python
MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')
```

- `pool_size`: An integer or tuple/list of 2 integers, window size over which to take the maximum.
- `strides`: An integer or tuple/list of 2 integers, specifying the strides of the pooling operation.
- `padding`: One of 'valid' or 'same' (case-insensitive).

**Dropout Layers:**

- Used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.

```Python
Dropout(rate)
```

- `rate`: Float between 0 and 1, fraction of the input units to drop.

## Recurrent Networks in Keras

Recurrent Neural Networks (RNNs) are a class of neural networks that are effective for modeling sequence data such as time series or natural language. Keras provides several layers for building RNNs, including the simple RNN layer, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) layers. These layers can be easily incorporated into a `Sequential` model or a more complex model architecture.

**NOTE:** RNNs can be challenging to train due to issues like vanishing and exploding gradients, and techniques like gradient clipping can be useful.

**SimpleRNN**

- **Description**: The `SimpleRNN` layer is a fully-connected RNN where the output from the previous timestep is to be fed to the next timestep.
- **Usage Example**:
    
    ```Python
    from keras.layers import SimpleRNN
    
    model.add(SimpleRNN(units=50, activation='tanh', return_sequences=True))
    
    ```
    
    - `units`: Positive integer, dimensionality of the output space.
    - `activation`: Activation function to use (default is 'tanh').
    - `return_sequences`: Boolean. Whether to return the last output in the output sequence, or the full sequence.

**LSTM**

- **Description**: LSTM, or Long Short-Term Memory, layers are a type of RNN layer that helps avoid the vanishing gradient problem and is capable of learning long-term dependencies.
- **Usage Example**:
    
    ```Python
    from keras.layers import LSTM
    
    model.add(LSTM(units=50, return_sequences=False))
    
    ```
    
    - `units`: Positive integer, dimensionality of the output space.
    - `return_sequences`: Boolean. Whether to return the last output or the full sequence.

**GRU**

- **Description**: GRU, or Gated Recurrent Unit, layers are a variant of LSTM with a simpler structure and can perform comparably to LSTM for many tasks.
- **Usage Example**:
    
    ```Python
    from keras.layers import GRU
    
    model.add(GRU(units=50, return_sequences=False))
    
    ```
    
    - `units`: Positive integer, dimensionality of the output space.
    - `return_sequences`: Boolean. Whether to return the last output or the full sequence.

## Incorporating Recurrent Layers in a Model

Recurrent layers can be used in a `Sequential` model just like any other layer. They are particularly useful for processing sequences of data, such as time series data or text. For sequence prediction tasks, it's common to set `return_sequences=True` for all recurrent layers except the last one, allowing the model to make a prediction based on the entire input sequence.

### Example: Building a Sequential Model with Recurrent Layers

Here's an example of how to build a simple RNN model for sequence processing using the `Sequential` API in Keras:

```Python
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  # Last layer only returns the last outputs
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

```

In this example, the `Embedding` layer is used to encode input sequences into dense vectors of fixed size. The `SimpleRNN` layers process these sequences, and the `Dense` layer outputs the final prediction. This setup is typical for tasks like sentiment analysis, where the model needs to understand the entire sequence to make a prediction.

Recurrent layers in Keras offer a straightforward way to build powerful models for sequence data, with flexibility to fit a wide range of tasks and data types.

## Common Methods

- `**model.compile(optimizer, loss, metrics)**`:
    - Configures the model for training.
    - `optimizer`: String (name of optimizer) or optimizer instance. Examples: 'adam', 'sgd'.
    - `loss`: String (name of objective function) or objective function. Examples: 'categorical_crossentropy', 'mean_squared_error'.
    - `metrics`: List of metrics to be evaluated by the model during training and testing. Example: `['accuracy']`.
- `**model.fit(x, y, batch_size, epochs, validation_data)**`:
    - Trains the model for a fixed number of epochs (iterations on a dataset).
    - `x`: Input data.
    - `y`: Target data.
    - `batch_size`: Number of samples per gradient update.
    - `epochs`: Number of epochs to train the model.
    - `validation_data`: Data on which to evaluate the loss and any model metrics at the end of each epoch.
- `**model.evaluate(x, y, batch_size)**`:
    - Returns the loss value & metrics values for the model in test mode.
    - `x`: Input data.
    - `y`: Target data.
    - `batch_size`: Number of samples per evaluation step.
- `**model.predict(x, batch_size)**`:
    - Generates output predictions for the input samples.
    - `x`: Input data.
    - `batch_size`: Number of samples per prediction step.

Understanding these layers and methods will provide a solid foundation for building a wide variety of neural network architectures in Keras.

## Pre-trained Models

Keras offers a suite of pre-defined and pre-trained models that can significantly accelerate the development process, especially in the domain of image recognition and classification. Among these, ResNet, VGG, and Inception stand out due to their proven effectiveness and widespread use.

### ResNet

- **Overview**: ResNet, short for Residual Networks, introduces a novel architecture with "skip connections" or "shortcut connections" allowing it to effectively train very deep networks.
- **Use Cases**: ResNet shines in tasks requiring deep networks, like complex image recognition and classification challenges.
- **Example Initialization**:
    
    ```Python
    from keras.applications.resnet50 import ResNet50
    model = ResNet50(weights='imagenet')
    
    ```
    
- **Key Points**:
    - `weights='imagenet'` loads the model pre-trained on the ImageNet dataset, offering a strong feature extractor out of the box.
    - ResNet variants (e.g., ResNet50, ResNet101, ResNet152) differ in depth, allowing flexibility based on computational resources and task complexity.

### VGG

- **Overview**: VGG, developed by the Visual Graphics Group at Oxford, is known for its simplicity, using only 3x3 convolutional layers stacked on top of each other in increasing depth.
- **Use Cases**: Ideal for image classification and localization tasks, VGG provides excellent generalization capabilities.
- **Example Initialization**:
    
    ```Python
    from keras.applications.vgg16 import VGG16
    model = VGG16(weights='imagenet')
    
    ```
    
- **Key Points**:
    - Similar to ResNet, `weights='imagenet'` indicates using a pre-trained model, making VGG an effective tool for tasks with similar data distributions to ImageNet.
    - VGG models (e.g., VGG16, VGG19) offer choices in depth, trading off between performance and computational efficiency.

### Inception

- **Overview**: The Inception architecture, starting from Inception v1 (or GoogleNet), introduces modules with parallel convolutional layers, significantly increasing the network's width.
- **Use Cases**: Inception excels in classifying and detecting objects within images, capable of focusing on local features with varying resolutions.
- **Example Initialization**:
    
    ```Python
    from keras.applications.inception_v3 import InceptionV3
    model = InceptionV3(weights='imagenet')
    
    ```
    
- **Key Points**:
    - The use of mixed convolutional layers allows the model to capture information at various scales effectively.
    - Later versions, like InceptionV3 and Inception-ResNet, incorporate advancements from both architectures, offering improved accuracy and efficiency.

These pre-defined models in Keras not only provide a solid foundation for numerous image processing tasks but also serve as educational tools, allowing a deeper understanding of successful neural network architectures. Leveraging these models can either be through direct application, fine-tuning for specific tasks, or as a feature extractor in a larger model pipeline.