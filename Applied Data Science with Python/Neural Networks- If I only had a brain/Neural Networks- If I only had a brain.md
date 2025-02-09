[Lecture 6 Overview](https://www.notion.so/Lecture-6-0a23c37715414365bf7d3c1ffa885aa0?pvs=21)

# Before we begin

![[Screenshot_2024-02-14_at_8.32.21_PM.png]]

1. No more homework! [[Project]]
    
    > Fun example: 20 Questions bot [https://github.com/earthtojake/20q](https://github.com/earthtojake/20q)
    
2. Tips for using the shell (terminal, command line)
    
    ![[ShellIntro.pdf]]
    

  

# **Neural networks overview**

![[70wnk5kfr5hc1.jpeg]]

## Biological inspiration

![[Notion/Getting into Data Science/Applied Data Science with Python/Neural Networks- If I only had a brain/Untitled.png|Untitled.png]]

A **neuron** has:

- Branching input (dendrites)
- Branching output (the axon)

The information circulates from the dendrites to the axon via the cell body

Axon connects to dendrites via synapses

- Synapses vary in strength
- Synapses may be excitatory or inhibitory

### _Pigeons as art experts_ (Watanabe _et al._ 1995)

Experiment:

- Pigeon in Skinner box
- Present paintings of two different artists (e.g. Chagall / Van Gogh)
- Reward for pecking when presented a particular artist (e.g. Van Gogh)

![[Notion/Getting into Data Science/Applied Data Science with Python/Neural Networks- If I only had a brain/Untitled 1.png|Untitled 1.png]]

![[Notion/Getting into Data Science/Applied Data Science with Python/Neural Networks- If I only had a brain/Untitled 2.png|Untitled 2.png]]

![[Notion/Getting into Data Science/Applied Data Science with Python/Neural Networks- If I only had a brain/Untitled 3.png|Untitled 3.png]]

Pigeons were able to discriminate between Van Gogh and Chagall with 95% accuracy (when presented with pictures they had been trained on). Discrimination still 85% successful for previously unseen paintings of the artists

Pigeons do not simply memorize the pictures!

- They can extract and recognize patterns (the ‘style’)
- They generalize from the already seen to make predictions

This is what neural networks (biological and artificial) are good at (unlike conventional computer)

## Artificial neural networks

Neural networks draw inspiration from the biological neural networks that constitute animal brains. Just as biological neurons transmit signals to each other via synapses, artificial neural networks (ANNs) consist of interconnected nodes or "neurons" that process and pass on information. This design allows ANNs to learn and make decisions, mimicking some level of natural intelligence.

**Artificial neurons:** Non-linear, parameterized function with restricted output range

![[Screenshot_2024-02-26_at_1.02.39_PM.png]]

![[ann.png]]

![[nn_overview.png]]

## Famous application: tank or not-a-tank

In the 1980s, the Pentagon wanted to harness computer technology to make their tanks harder to attack.

The preliminary plan was to fit each tank with a digital camera hooked up to a computer. The computer would continually scan the environment outside for possible threats - such as an enemy tank hiding behind a tree - and alert the tank crew to anything suspicious.

Computers are really good at doing repetitive tasks without taking a break, but they are generally bad at interpreting images. The only possible way to solve the problem was to employ a neural network.

The research team went out and took 100 photographs of tanks hiding behind trees, and then took 100 photographs of trees - with no tanks. They took 50 photos from each group and put them in a vault for safe-keeping. They scanned the remaining 100 photos into their mainframe computer.

![[Notion/Getting into Data Science/Applied Data Science with Python/Neural Networks- If I only had a brain/Untitled 4.png|Untitled 4.png]]

![[Notion/Getting into Data Science/Applied Data Science with Python/Neural Networks- If I only had a brain/Untitled 5.png|Untitled 5.png]]

### Success!

The huge neural network was fed each photo one at a time and asked if there was a tank hiding behind the trees. Of-course at the beginning its answers were completely random since the network didn't know what was going on or what it was supposed to do. But each time it was fed a photo and it generated an answer, the scientists told it if it was right or wrong. If it was wrong it would randomly change the weightings in its network until it gave the correct answer.

But the scientists were worried: _had it actually found a way to recognize if there was a tank in the photo, or had it merely memorized which photos had tanks and which did not?_

This is a big problem with neural networks, after they have trained themselves you have no idea how they arrive at their answers, they just do. The question was did it understand the concept of tanks vs. no tanks, or had it merely memorized the answers? So the scientists took out the photos they had been keeping in the vault and fed them through the computer. The computer had never seen these photos before - this would be the big test. **To their immense relief the neural net correctly identified each photo as either having a tank or not having one.**

### Testing with new data

The Pentagon was very pleased with this, but a little bit suspicious, they wanted to see this marvel of modern technology for themselves. They took another set of photos (half with tanks and half without) and scanned them into the computer and through the neural network.

**The results were completely random**. For a long time nobody could figure out why. After all nobody understood how the neural had trained itself.

**The military was now the proud owner of a multi-million dollar mainframe computer that could tell you if it was sunny or not!**

## Applications in Machine Learning

Neural networks have revolutionized the field of machine learning, providing the backbone for a myriad of applications:

- **Image Recognition:** ANNs, particularly Convolutional Neural Networks (CNNs), have become instrumental in image analysis, powering applications from facial recognition systems to medical imaging diagnostics.
- **Natural Language Processing (NLP):** Through models like Recurrent Neural Networks (RNNs) and more recently, Transformers, neural networks have significantly advanced the ability of computers to understand and generate human language, enabling technologies such as language translation services, chatbots, and voice-activated assistants.
- **Autonomous Driving:** Neural networks are at the heart of autonomous vehicle systems, enabling them to interpret sensor data, make decisions, and learn from vast amounts of driving data to navigate safely.

## Simulating Complex Functions

One of the most profound aspects of neural networks is their ability to approximate virtually any complex function, a property known as the **Universal Approximation Theorem**. This theorem suggests that a feedforward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of $\mathbb{R}^n$﻿, given appropriate activation functions.

The **layered composition** of neural networks, where each layer's output serves as the input to the next, allows these models to learn hierarchies of features. In the context of image recognition, for instance, initial layers might learn to recognize edges and basic textures, while deeper layers can identify more complex structures like shapes or specific objects. This hierarchical learning makes neural networks particularly adept at handling data with complex, hierarchical structures, such as images, sound, and text.

![[approximation.png]]

# **Activation functions**

Activation functions play a crucial role in neural networks by introducing non-linearity. Without these functions, a neural network, regardless of its depth, would essentially behave like a linear model, unable to capture the complex patterns found in real-world data. Non-linearity allows neural networks to learn and model complex relationships between input and output data, making them capable of performing tasks like image recognition, language translation, and many others beyond the scope of simple linear models.

In essence, activation functions enable neural networks to solve non-linear problems, expanding their applicability far beyond linear models. Coupled with careful input preparation, neural networks can model complex functions and discover intricate patterns in vast and varied datasets.

## Similarity to Logistic Regression

The concept of activation functions in neural networks bears a resemblance to logistic regression in several ways:

- **Weighted Sum Inputs:** Both neural networks and logistic regression models compute a weighted sum of the input features. In neural networks, this sum is then passed through an activation function.
- **Activation Output:** The activation function's output can be seen as a decision, similar to the logistic function in logistic regression, which maps the weighted sum (plus bias term) to a probability score indicating the likelihood of a particular class or outcome. In short, the output is always a score in the interval $y \in [0,1] = f(\{x_i\}) = \sum{w_i x_i} + b$﻿

## Introducing: ReLU

The **Rectified Linear Unit (ReLU)** has become one of the most widely used activation functions in neural networks, especially in deep learning architectures. ReLU is defined as $f(x) = \max(0,x)$﻿, effectively replacing all negative values in the activation map with zero.

**Note:** There are other common activation functions, including sigmoid, tanh, and Leaky ReLU

![[relu.png]]

### Advantages of ReLU:

Its popularity stems from its simplicity and efficiency, offering several advantages:

- **Simplicity:** The ReLU function is computationally efficient, allowing neural networks to train faster and perform better, especially in deep learning architectures. Its simple max operation is much less computationally expensive than functions like sigmoid or tanh.
- **Mitigating Vanishing Gradient Problem:** ReLU helps in mitigating the vanishing gradient problem, which is prevalent in deep networks with saturating activation functions. Since the gradient of ReLU for positive inputs is always 1, it ensures that the gradient does not vanish during backpropagation, facilitating deeper network training.
- **Sparse Activation:** In ReLU, only positive values have non-zero outputs, leading to sparse activations within neural networks. This sparsity can lead to more efficient and less overfitting-prone models, as not all neurons are activated simultaneously.
- **Improved Gradient Flow:** For positive input values, the derivative of ReLU is constant (1), which means that the gradient flow during backpropagation is not hindered by the activation function. This allows for more effective learning, especially in deeper layers of a network.

Overall, ReLU's introduction marked a significant advancement in neural network activation functions, contributing to the rapid development of deep learning by enabling the training of much deeper networks than was previously feasible.

# Preparing Inputs

Proper input preparation is crucial for the efficient and effective training of neural networks.

Typically, the initial layer of a neural network assigns individual neurons to specific inputs, which, with the **Universal Approximation Theorem**, provides a degree of resilience to inputs that vary widely in scale. However, preparing inputs through meticulous cleaning, transformation, and feature engineering remains vital. This preparation streamlines the problem the network must solve by consolidating co-linear inputs, harmonizing scales, and merging inputs that might have significant interactions.

> [!important]  
> Consider the neurons as a finite resource: data preparation can spare capacity that would otherwise be used to approximate these preprocessing steps  

## Data cleaning & transformation

- **Normalization:** Scaling input features so they are on a similar scale can prevent certain features from dominating due to their scale. Normalization adjusts the data to fall within a smaller, specified range, such as -1 to 1 or 0 to 1.
- **Standardization:** This involves transforming the data to have a mean of zero and a standard deviation of one. Standardization ensures that the feature distribution is centered around 0, with a standard deviation that scales the distribution. This is particularly useful for inputs to activation functions that are sensitive to magnitude, such as sigmoid or tanh.
- **Handling Missing Values:** Missing data can significantly impact the performance of neural networks. Techniques such as imputation (filling missing values with the mean, median, or mode), or using a model to predict missing values, can be employed to address this issue.

## Input Shape Importance

- **Consistent Dimensions:** Neural networks require a fixed size of input; thus, it's crucial to preprocess the data to ensure consistent dimensions. For images, this might involve cropping or padding to achieve uniform dimensions. For text or sequences, this could mean padding shorter sequences or truncating longer ones to a fixed length.
- **Batch Size:** The choice of batch size can affect both the speed and stability of the training process. Larger batches provide a more accurate estimate of the gradient, but they require more memory and might lead to slower convergence.

## Feature Engineering

- **Feature Selection:** Identifying and selecting the most informative features can reduce the dimensionality of the data and improve model performance. Techniques such as correlation analysis, principal component analysis (PCA), or model-based selection can be used to identify the most relevant features.
- **Feature Encoding:** Proper encoding of categorical variables is crucial. Techniques like one-hot encoding or embedding layers for deep learning models can transform categorical variables into a format that neural networks can work with effectively.
- **Feature Construction:** Creating new features through domain knowledge or by combining existing features can provide additional information to the model, potentially improving its performance. For example, creating polynomial features or interaction terms might expose new patterns to the model.
- **Temporal and Spatial Features:** For time series data, deriving features like rolling averages or time lags can capture temporal dynamics. For spatial data, features that capture spatial relationships or clustering can be beneficial.

# **Training Neural Networks**

Training neural networks involves adjusting the weights and biases of the network to minimize the difference between the predicted output and the actual output. This process is guided by several key components and techniques:

## Backpropagation

**Backpropagation** is the cornerstone of neural network training, allowing the adjustment of weights in the network based on the error rate obtained in the previous epoch (i.e., iteration). It effectively distributes the error back through the network layers, providing insight into the responsibility of each weight towards the error.

![[backpropagation.png]]

## Gradient descent

**Gradient descent** is the optimization algorithm used to minimize the cost function, which represents the difference between the network's predicted output and the actual output. By calculating the gradient of the cost function, gradient descent adjusts the weights in the direction that most reduces the cost.

## Cost Functions

In neural network training, selecting an appropriate cost function is crucial as it guides the optimization process.

Each cost function has its specific use cases and considerations. The choice depends on the particular problem, the type of neural network being trained, and the desired properties of the model (e.g., robustness to outliers, probabilistic output).

Here are some commonly used cost functions along with their strengths and weaknesses:

- **Mean Squared Error (MSE):**
    - **Strengths:** Intuitive, widely used for regression tasks; heavily penalizes large errors due to squaring, leading to robust models.
    - **Weaknesses:** Can be overly sensitive to outliers; assumes a Gaussian distribution of errors.
- **Cross-Entropy Loss:**
    - **Strengths:** Ideal for classification tasks; well-suited for models outputting probabilities (e.g., models with a softmax final layer).
    - **Weaknesses:** Can lead to numerical instability if not implemented with care (e.g., log(0) situations).
- **Binary Cross-Entropy Loss:**
    - **Strengths:** Special case of cross-entropy for binary classification tasks; aligns well with models outputting a probability between 0 and 1.
    - **Weaknesses:** Not suitable for multi-class classification tasks.
- **Hinge Loss:**
    - **Strengths:** Commonly used for Support Vector Machines and "maximum-margin" classification, encouraging examples to be on the correct side of the margin.
    - **Weaknesses:** Not as interpretable as probabilistic losses like cross-entropy; less common in neural networks.
- **Kullback-Leibler Divergence (KL Divergence):**
    - **Strengths:** Measures how one probability distribution diverges from a second, expected distribution; useful in unsupervised learning, reinforcement learning, and models like autoencoders.
    - **Weaknesses:** Asymmetric, which can be a limitation depending on the application; requires careful handling of zero probabilities.
- **Huber Loss:**
    - **Strengths:** Combines the best of MSE and absolute loss by being quadratic for small errors and linear for large errors, reducing sensitivity to outliers compared to MSE.
    - **Weaknesses:** The transition between quadratic and linear (controlled by the δ parameter) can be arbitrary and may need tuning.
- **Log-Cosh Loss:**
    - **Strengths:** Smooth approximation of the MSE that remains numerically stable; behaves like MSE for small errors and like absolute loss for large errors.
    - **Weaknesses:** Not as commonly used or understood as MSE or cross-entropy; may require more computational resources due to the use of hyperbolic cosine function.

### **Advanced Optimization Techniques**

Beyond basic gradient descent, there are several optimization algorithms like SGD (Stochastic Gradient Descent), Adam (Adaptive Moment Estimation), and RMSprop (Root Mean Square Propagation), each with its own advantages in terms of speed and convergence stability.

## Evaluation Metrics

Evaluation metrics are crucial for assessing the performance of neural networks.

Each metric offers unique insights into model performance, and the choice of metric should align with the specific objectives and constraints of the task at hand. In practice, it's often beneficial to consider multiple metrics to gain a comprehensive understanding of a model's strengths and weaknesses.

- **Accuracy:**
    - **Strengths:** Intuitive and straightforward; measures the proportion of correct predictions.
    - **Weaknesses:** Can be misleading in imbalanced datasets where one class dominates.
- **Precision and Recall:**
    - **Strengths:** Useful in imbalanced datasets; precision focuses on the quality of positive predictions, while recall emphasizes the coverage of actual positive cases.
    - **Weaknesses:** Trade-off between the two (improving one can worsen the other); doesn't provide a single metric for optimization.
- **F1 Score:**
    - **Strengths:** Harmonic mean of precision and recall, providing a single metric that balances the two; useful in imbalanced datasets.
    - **Weaknesses:** May not capture the nuances in cases where one aspect (precision or recall) is more important than the other.
- **Mean Squared Error (MSE) and Root Mean Squared Error (RMSE):**
    - **Strengths:** Directly corresponds to the cost function used in many regression tasks; easy to interpret in terms of the data scale.
    - **Weaknesses:** Highly sensitive to outliers; may not accurately reflect performance in non-Gaussian distributions.
- **Mean Absolute Error (MAE):**
    - **Strengths:** Intuitive, represents average error magnitude without considering direction; less sensitive to outliers than MSE.
    - **Weaknesses:** May not fully capture the impact of large errors as MSE does.
- **Area Under the ROC Curve (AUC-ROC):**
    - **Strengths:** Represents model's ability to discriminate between classes; robust to imbalanced datasets; can be used to choose decision thresholds.
    - **Weaknesses:** May be less informative in highly imbalanced situations or when different costs are associated with different types of errors.
- **Confusion Matrix:**
    - **Strengths:** Provides a detailed breakdown of predictions vs. actual values, allowing for in-depth analysis of type I and type II errors.
    - **Weaknesses:** More complex to interpret at a glance than a single metric; doesn't summarize performance into a single number.
- **Log Loss (for classification):**
    - **Strengths:** Penalizes confidence in wrong predictions; useful for probabilistic outputs.
    - **Weaknesses:** Can be heavily influenced by small probabilities; less intuitive than accuracy or error rate.

## Regularization and Overfitting

Overfitting occurs when a model learns the training data too well, including its noise, resulting in poor performance on unseen data. Techniques to combat overfitting include simplifying the model, using more training data, and employing regularization techniques.

**Regularization methods** like L1 and L2 regularization, dropout, and early stopping add constraints to the network or its training process to prevent overfitting by discouraging overly complex models.

### **Ethics of Overfitting**

Overfitting can also lead to biased models, especially if the training data is not representative of the general population. Ethical considerations necessitate careful examination of the data and model to ensure biases are not perpetuated.

## Monitoring the training process with TensorBoard

In our examples we monitor training via text output, but more sophisticated tools exist to visualize the training process. One popular tool is **TensorBoard**, which can be useful when training large models using parallelization across multiple GPUs.

Incorporating TensorBoard into the training process not only aids in model development and tuning but also enhances transparency and understanding of the model's learning dynamics, making it an invaluable tool in the neural network training toolkit.

- **Real-time Monitoring:** TensorBoard provides a user-friendly interface to monitor the training process in real time, allowing for the visualization of metrics like loss and accuracy across epochs, which is crucial for understanding model performance and convergence.
- **Hyperparameter Tuning:** It offers tools for hyperparameter tuning, enabling the comparison of model performance across different sets of hyperparameters, which is essential for optimizing model configurations.
- **Model Architecture Visualization:** TensorBoard can visualize the neural network's architecture, offering insights into the model's structure and helping identify potential areas for improvement or optimization.
- **Gradient and Weight Visualization:** It allows for the inspection of gradients and weights during training, helping to diagnose issues related to learning, such as vanishing or exploding gradients.
- **Embedding Visualization:** TensorBoard provides functionalities to visualize high-dimensional data embeddings, which can be particularly useful for tasks involving complex data representations, such as NLP or image processing.

## Interpretability and Explainability

Interpretability and explainability are crucial for understanding how machine learning models, especially neural networks, arrive at their predictions.

**Interpretability** refers to the degree to which a human can understand the cause of a decision, and **explainability** involves the clarity with which a model can describe its functioning. Techniques for enhancing these aspects include feature importance, model simplification, and visualization tools. Ensuring models are interpretable and explainable is crucial, especially in sensitive applications like healthcare and finance, where decisions need to be justified and understood by stakeholders.

When selecting tools for interpretability and explainability, it's essential to consider the model type, the complexity of the dataset, the computational resources available, and the specific needs of the stakeholders who will be using the explanations. Combining multiple approaches can often provide a more comprehensive understanding of the model's behavior.

### Tools and Libraries for Interpretability and Explainability

- **LIME (Local Interpretable Model-agnostic Explanations):**
    - **Features:** Generates explanations for individual predictions, showing how different features influence the output.
    - **Strengths:** Model-agnostic; can be used with any model type.
    - **Weaknesses:** Local explanations may not provide a complete picture of the model's overall behavior.
- **SHAP (SHapley Additive exPlanations):**
    - **Features:** Uses game theory to explain the output of any model by computing the contribution of each feature to the prediction.
    - **Strengths:** Considers feature interactions; provides both local and global explanations.
    - **Weaknesses:** Can be computationally expensive, especially for complex models and large datasets.
- **Feature Importance:**
    - **Features:** Ranks features based on their importance in the model, often derived from the model itself (e.g., coefficients in linear models, feature importance in tree-based models).
    - **Strengths:** Provides a straightforward, global view of feature relevance.
    - **Weaknesses:** May not capture nonlinear relationships or interactions between features effectively.
- **Partial Dependence Plots (PDPs) and Individual Conditional Expectation (ICE) Plots:**
    - **Features:** PDPs show the average effect of a feature on the prediction, while ICE plots show this effect for individual instances.
    - **Strengths:** Offers insights into the model's behavior over a range of feature values.
    - **Weaknesses:** PDPs can be misleading if features are correlated; ICE plots can become cluttered with many instances.
- **Integrated Gradients:**
    - **Features:** Attribute the prediction of a deep network to its input features, based on gradients.
    - **Strengths:** Provides detailed explanations suitable for complex models like deep neural networks.
    - **Weaknesses:** Interpretation of the results can be challenging, especially with high-dimensional data.
- **Counterfactual Explanations:**
    - **Features:** Explains model decisions by showing how slight changes in input features could lead to different predictions.
    - **Strengths:** Intuitive and actionable insights; user-friendly explanations.
    - **Weaknesses:** Generating relevant and realistic counterfactuals can be complex.
- **Grad-CAM (Gradient-weighted Class Activation Mapping):**
    - **Features:** Uses gradients flowing into the final convolutional layer of CNNs to produce a heatmap highlighting important regions in the input image for predicting the concept.
    - **Strengths:** Offers visual explanations that are easy to interpret.
    - **Weaknesses:** Specific to convolutional neural networks; may not be applicable to other model types.

# **Model Architecture**

The architecture of a neural network is a critical factor that defines its ability to learn and solve complex problems. It encompasses the layout of neurons and layers, how they're interconnected, and the flow of data through the network. This section explores various architectural designs, their unique features, and their suitability for different tasks in machine learning.

## Network Depth & Connectedness

### Shallow vs. Deep

- **Shallow Networks:** Typically consist of a few layers, including input and output layers, and perhaps one or two hidden layers. Shallow networks are suited for simpler problems where the relationship between the input and output is not overly complex.
- **Deep Networks:** Contain many layers, sometimes hundreds or thousands, enabling them to learn features at multiple levels of abstraction. Deep networks are more suited for complex problems like image recognition, where higher-level features (like shapes) are built from lower-level features (like edges and corners).

### Depth Challenges

- Deep networks can be more challenging to train due to issues like vanishing and exploding gradients. Advanced techniques like residual connections (ResNets), batch normalization, and advanced optimizers have been developed to address these challenges, allowing for successful training of deep networks.

### Connectedness: Dense vs. Sparse

The connectedness of a neural network refers to how neurons within layers are linked to each other and to neurons in adjacent layers. This structure significantly influences the network's capacity to capture patterns and relationships in the data.

- **Fully Connected Layers:** In a fully connected (or dense) layer, every neuron is connected to every neuron in the previous and following layers. This setup is powerful for capturing complex relationships but can be computationally expensive and prone to overfitting, especially in deep networks with a large number of parameters.
- **Sparse Connectivity:** To reduce computational demands and overfitting, some architectures employ sparse connectivity, where neurons are only connected to a subset of neurons in adjacent layers. Convolutional layers in CNNs are an example, where each neuron is connected only to a local region of the input.
- **Skip Connections:** An innovation to improve training in deep networks, skip connections, as used in ResNet architectures, allow the gradient to flow directly through the network by skipping one or more layers. This design helps mitigate the vanishing gradient problem and supports the training of very deep networks.
- **Recurrent Connections:** In RNNs, connections between neurons form loops, creating a 'memory' of previous inputs. This structure is ideal for sequential data, allowing the network to maintain information across time steps, which is crucial for tasks like language modeling and time series prediction.

The connectedness within a neural network's architecture is pivotal in defining its learning capabilities, computational efficiency, and applicability to different tasks. By carefully designing the network's structure, it's possible to balance the model's expressiveness with its generalizability and computational demands.

## Convolution and Recurrence: Architectures for Specific Data Types

By understanding the unique characteristics and strengths of CNNs and RNNs, neural network designers can choose or craft architectures that are best suited to the specific requirements of their machine learning tasks, whether they involve analyzing visual data, decoding language, predicting future events, or any other application that relies on understanding spatial or temporal data.

### Convolutional Neural Networks (CNNs)

CNNs are specialized for processing data with a known grid-like structure, exemplified by image data.

- **Convolutional Layers:** The core building blocks of CNNs, these layers apply a convolution operation to the input, passing the result to the next layer. This process allows the network to build a complex hierarchy of features from simple patterns to more abstract concepts, making CNNs highly effective for tasks like image and video recognition, image classification, and more.
- **Pooling Layers:** Often follow convolutional layers and are used to reduce the spatial dimensions (width and height) of the input volume for the next convolutional layer. Pooling helps in reducing the number of parameters and computation in the network, and hence also helps in controlling overfitting.

### Recurrent Neural Networks (RNNs)

RNNs are designed to handle sequential data, where the order of the data points is significant.

- **Sequence Processing:** Unlike feedforward neural networks, RNNs have loops in them, allowing information to persist. This architecture makes them ideal for tasks where context from previous inputs is crucial, such as in language modeling or time series prediction.
- **Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU):** Variants of RNNs designed to solve the vanishing gradient problem associated with standard RNNs. LSTMs and GRUs introduce gates that regulate the flow of information, allowing the network to retain or forget information over long sequences effectively.
    
    ![[rnn.png]]
    

![[LSTMvsGRU.png]]

![[lstmvsgru2.png]]

### Architectural Considerations

- **Spatial vs. Temporal Data:** CNNs excel at capturing spatial hierarchies in data (like images), where the location of features within the data is key. RNNs, on the other hand, shine with temporal data (like text or time series), where the sequence of data points is crucial.
- **Parameter Sharing:** CNNs use parameter sharing (the same filter applied across the image), which significantly reduces the number of parameters in the network, making them computationally efficient. RNNs share parameters across time steps, allowing them to process sequences of any length.
- **Applicability:** The choice between CNNs and RNNs (and their variants) depends heavily on the nature of the problem at hand. For mixed data types or complex tasks, **hybrid models** that combine aspects of CNNs and RNNs might be used to leverage the strengths of both architectures.

## Transformers and Attention

**Transformers** have redefined the landscape of neural network architectures, particularly in the field of Natural Language Processing (NLP) and beyond. By introducing a novel structure that leverages the power of attention mechanisms, transformers offer a significant departure from traditional recurrent models.

The first appearance of transformers is in the paper [**Attention is All You Need**](https://arxiv.org/abs/1706.03762), published by researchers at Google.

Transformers have rapidly become the architecture of choice for a wide range of NLP tasks, achieving state-of-the-art results in machine translation, text generation, sentiment analysis, and more. Their flexibility and efficiency have also inspired adaptations of the transformer architecture to other domains, such as computer vision and audio processing, marking a significant evolution in the field of deep learning.

![[tx_basic.png]]

![[tx_moderate.png]]

![[1_vrSX_Ku3EmGPyqF_E-2_Vg.png]]

### Transformer Architecture

- **Parallel Processing:** Unlike their recurrent predecessors, transformers process entire sequences simultaneously, which eliminates the sequential computation inherent in RNNs and LSTMs. This characteristic allows for substantial improvements in training efficiency and model scalability.
- **Self-Attention:** At the heart of the transformer architecture is the self-attention mechanism, which computes the representation of a sequence by relating different positions of a single sequence. This mechanism enables the model to dynamically weigh the importance of each part of the input data, enhancing its ability to capture complex relationships within the data.
- **Layered Structure:** Transformers are composed of stacked layers of self-attention and position-wise feedforward networks. Each layer in the transformer processes the entire input data in parallel, which contributes to the model's exceptional efficiency and effectiveness.

### Attention Mechanism

The **attention** mechanism allows transformers to consider the entire context of the input sequence, or any subset of it, regardless of the distance between elements in the sequence. This global view is particularly advantageous for tasks that require understanding long-range dependencies, such as document summarization or question-answering.

![[attention.png]]

The left and center figures represent different layers / attention heads. The right figure depicts the same layer/head as the center figure, but with the token _lazy_ selected

![[simple-pretty-gif.gif]]

- **Scaled Dot-Product Attention:** The most commonly used attention mechanism in transformers involves computing the dot product of the query with all keys, dividing each by the square root of the dimension of the keys, applying a softmax function to obtain the weights on the values. This approach efficiently captures the relevance of different parts of the input data to each other.
- **Multi-Head Attention:** Transformers further extend the capabilities of the attention mechanism through the use of multi-head attention. This involves running multiple attention operations in parallel, with each "head" focusing on different parts of the input data. This diversity allows the model to attend to different aspects of the data, enhancing its representational power.

## Specialized Architectures

- **Generative Adversarial Networks (GANs)** consist of two networks, a generator and a discriminator, that are trained simultaneously. The generator learns to generate data similar to the input data, while the discriminator learns to distinguish between the generated data and the real data.
- **Graph Neural Networks (GNNs)** extend neural network methods to graph data, enabling the modeling of relationships and interactions in data structured as graphs. They are particularly useful in social network analysis, chemical molecule study, and recommendation systems.
- **Capsule Networks** offer an alternative to traditional convolutional networks by grouping neurons into “capsules” that represent various properties of the same entity, allowing the network to learn part-whole relationships. This architecture is designed to preserve spatial hierarchies between features, making it beneficial for tasks that require a high level of interpretability, such as object detection and recognition in images
- **Autoencoders** are designed for unsupervised learning tasks, such as dimensionality reduction or feature learning. They work by compressing the input into a latent-space representation and then reconstructing the output from this representation.
- **Variational Autoencoders (VAEs)** are generative models similar to autoencoders that learn to encode data into a latent space and reconstruct it. However, VAEs introduce a probabilistic twist, modeling the latent space as a distribution, which allows for the generation of new data points by sampling from this space
- **Diffusion Models** gradually add noise to data until it becomes indistinguishable from random noise and then learn to reverse this noising process to generate data. While they don't use a latent space in the traditional sense, the intermediate noisy states during the reverse process can be viewed as a form of high-dimensional latent representation.

## Interlude: Latent Space

The concept of latent space is particularly relevant in the context of autoencoders and generative models within neural network architectures. You can introduce latent space in a subsection under these topics, explaining its significance in learning compact, meaningful representations of data. Here's a suggestion on where and how to incorporate it:

In the realm of autoencoders and generative models, such as Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs), the concept of latent space plays a central role. Latent space refers to the compressed representation that these models learn, which captures the essential information of the input data in a lower-dimensional form.

- **Role in Autoencoders:** In autoencoders, the encoder part of the model compresses the input into a latent space representation, and the decoder part attempts to reconstruct the input from this latent representation. The latent space thus acts as a bottleneck, forcing the autoencoder to learn the most salient features of the data.
- **Generative Model Applications:** In generative models like VAEs and GANs, the latent space representation can be sampled to generate new data points that are similar to the original data. For example, in the case of images, by sampling different points in the latent space, a model can generate new images that share characteristics with the training set but are not identical replicas.

### Significance of Latent Space

- **Data Compression:** Latent space representations allow for efficient data compression, reducing the dimensionality of the data while retaining its critical features. This aspect is particularly useful in tasks that involve high-dimensional data, such as images or complex sensor data.
- **Feature Learning:** The process of learning a latent space encourages the model to discover and encode meaningful patterns and relationships in the data, often leading to representations that can be useful for other machine learning tasks.
- **Interpretability and Exploration:** Examining the latent space can provide insights into the data's underlying structure. In some cases, latent space representations can be manipulated to explore variations of the generated data, offering a tool for understanding how different features contribute to the data generation process.

## Embeddings

**Embeddings** are a natural extension of the latent space, particularly in the context of handling high-dimensional categorical data. They transform sparse, discrete input features into a continuous, lower-dimensional space, much like the latent representations in autoencoders and generative models.

Embeddings map high-dimensional data, such as words or categorical variables, to a dense vector space where semantically similar items are positioned closely together. This transformation facilitates the neural network's task of discerning patterns and relationships in the data.

- **Application in NLP:** In the realm of Natural Language Processing, embeddings like Word2Vec and GloVe have transformed the way text is represented, enabling models to capture and utilize the semantic and syntactic nuances of language. Each word is represented as a vector, encapsulating its meaning based on the context in which it appears.
- **Categorical Data Representation:** Beyond text, embeddings are instrumental in representing categorical data in tasks beyond NLP. For example, in recommendation systems, embeddings can represent users and items in a shared vector space, capturing preferences and item characteristics that drive personalized recommendations.

### Advantages

- **Efficiency and Dimensionality Reduction:** Embeddings reduce the computational burden on neural networks by condensing high-dimensional data into more manageable forms without sacrificing the richness of the data's semantic and syntactic properties.
- **Enhanced Semantic Understanding:** By embedding high-dimensional data into a continuous space, neural networks can more easily capture and leverage the inherent similarities and differences within the data, leading to more accurate and nuanced predictions.
- **Facilitating Transfer Learning:** Similar to latent space representations, embeddings can be employed in a transfer learning context, where knowledge from one domain can enhance performance in related but distinct tasks.

## LLMs and the Rise of General-Purpose Models

Recent years have seen the emergence of large language models (LLMs) like GPT-3, BERT, and their successors, which represent a paradigm shift towards training massive, **general-purpose models**. These models are capable of understanding and generating human-like text and can be adapted to a wide range of tasks, from translation and summarization to question-answering and creative writing.

### Fine-Tuning

Training an existing transformer-based model on new data is called **fine-tuning**. It is one possible way to extend its capability.

- **Adaptability:** Fine-tuning involves taking a model that has been pre-trained on a vast corpus of data and adjusting its parameters slightly to specialize in a more narrow task. This process leverages the broad understanding these models have developed to achieve high performance on specific tasks with relatively minimal additional training.
- **Efficiency:** By starting with a pre-trained model, researchers and practitioners can bypass the need for extensive computational resources required to train large models from scratch. Fine-tuning allows for the customization of these powerful models to specific needs while retaining the general knowledge they have already acquired.

- Fine-tuning a GPT using PyTorch
    
    ### **Step 1: Install Transformers and PyTorch**
    
    Ensure you have the `**transformers**` and `**torch**` libraries installed. You can install them using pip if you haven't already:
    
    ```Shell
    pip install transformers torch
    ```
    
    ### **Step 2: Load a Pre-Trained Model and Tokenizer**
    
    First, import the necessary modules and load a pre-trained model along with its corresponding tokenizer. We'll use GPT-2 as an example.
    
    ```Python
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    ```
    
    ### **Step 3: Prepare Your Dataset**
    
    Prepare your text data for training. This involves tokenizing your text corpus and creating a dataset that the model can process.
    
    ```Python
    texts = ["Your text data", "More text data"]  # Replace with your text corpus
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    ```
    
    ### **Step 4: Fine-Tune the Model**
    
    Use the Hugging Face `**Trainer**` class or a custom training loop to fine-tune the model on your dataset. For simplicity, we'll use the `**Trainer**`.
    
    ```Python
    from transformers import Trainer, TrainingArguments
    
    training_args = TrainingArguments(
        output_dir="./results",           # Output directory
        num_train_epochs=3,               # Total number of training epochs
        per_device_train_batch_size=4,    # Batch size per device during training
        per_device_eval_batch_size=4,     # Batch size for evaluation
        warmup_steps=500,                 # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,                # Strength of weight decay
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs,  # Assuming `inputs` is your processed dataset
        # eval_dataset=eval_dataset,  # If you have a validation set
    )
    
    trainer.train()
    ```
    
    ### **Step 5: Save and Use the Fine-Tuned Model**
    
    After fine-tuning, save your model for future use and inference.
    
    ```Python
    pythonCopy code
    trainer.save_model("./fine_tuned_model")
    
    # To use the model for generation:
    prompt = "Your prompt here"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_text_ids = model.generate(input_ids, max_length=100)
    generated_text = tokenizer.decode(generated_text_ids[0], skip_special_tokens=True)
    
    print(generated_text)
    ```
    

### Prompt Engineering and One-Shot Learning

Prompt engineering is the art of crafting input prompts that guide the model to generate desired outputs. This technique exploits the model's ability to understand context and generate relevant responses, making it possible to "program" the model for new tasks without explicit retraining.

### **One-Shot and Few-Shot Learning**

One of the most remarkable capabilities of modern LLMs is their ability to perform tasks with minimal examples — sometimes just **one or a few (few-shot learning)**. This ability stems from their extensive pre-training, which provides a rich context for understanding and generating text.

# **Neural Networks in Practice**

## Areas of Active Research

The field of neural networks is vibrant with research activity, exploring both foundational theories and innovative applications:

- **Few-Shot Learning:** This research area focuses on designing models that can learn from a very limited amount of data, akin to human learning efficiency.
- **Generative Models:** Innovations in models like GANs and diffusion models are pushing the boundaries of content creation, from realistic images to synthetic data for training other models.
- **Reinforcement Learning:** Combining neural networks with reinforcement learning principles is leading to breakthroughs in autonomous systems, game playing, and decision-making processes.
- **Explainability and Ethics:** As neural networks become more integral to critical applications, understanding their decision-making process and ensuring ethical usage are paramount.

## Current Limitations

Despite significant advancements, neural networks still face several limitations:

- **Interpretability:** The "black-box" nature of deep neural networks makes it challenging to understand and trust their decisions, especially in critical applications.
- **Data Dependency:** High-performing neural networks often require vast amounts of labeled data, which can be expensive or infeasible to obtain for many problems.
- **Computational Resources:** Training state-of-the-art models requires significant computational power, often necessitating specialized hardware like GPUs or TPUs.
- **Hallucination:** A phenomenon where generative models, such as GPT or large-scale image generators, produce outputs that are plausible but factually incorrect or nonsensical. This issue is particularly prevalent in models trained on vast, uncurated datasets and can lead to misleading or false outputs.

### Addressing Hallucination

There is no general solution to preventing model hallucination. One way I like to think of it is akin to regression: when extrapolating beyond the training data you run the risk of making assumptions that no longer hold.

Approaches include:

- **Training Data Curation:** Carefully curating and vetting training datasets can reduce the likelihood of hallucination by ensuring that models learn from high-quality, accurate data.
- **Prompt and Output Design:** In generative models, carefully designing input prompts and setting constraints on outputs can mitigate hallucination effects. This is particularly relevant in NLP applications where the context and phrasing of prompts can significantly influence the model's output.
- **Human-in-the-loop:** Incorporating human feedback into the training loop can help identify and correct hallucinations, leading to models that better align with factual accuracy and user expectations.

  

## Design Patterns

To navigate the challenges and leverage the strengths of neural networks, practitioners have adopted several design patterns:

- **Ensemble Methods:** Combining predictions from multiple neural network models to improve overall performance and reduce the likelihood of overfitting. This approach is beneficial in competitions and critical applications where even minor performance improvements are valuable.
- **Attention Mechanisms:** Beyond their use in Transformers, attention mechanisms can enhance various neural network architectures by allowing models to focus on the most relevant parts of the input data, leading to better performance, especially in tasks involving sequences or contexts.
- **Normalization Techniques:** Batch normalization, layer normalization, and instance normalization are strategies to stabilize and accelerate neural network training. By normalizing the inputs to layers within the network, these techniques help mitigate issues like internal covariate shift.
- **Regularization Techniques:** Beyond L1/L2 regularization, techniques such as dropout, data augmentation, and early stopping are employed to prevent overfitting and ensure that models generalize well to unseen data.
- **Residual Connections:** Popularized by ResNet architectures, residual connections help alleviate the vanishing gradient problem in deep networks by allowing gradients to flow through skip connections, enabling the training of much deeper networks.
- **Dynamic Architectures:** Incorporating mechanisms that allow the network to adapt its structure or computation paths dynamically based on the input data, such as Neural Architecture Search (NAS) or conditional computation, can lead to more efficient and effective models.
- **Transfer Learning:** Leveraging pre-trained models on large datasets and fine-tuning them for specific tasks can significantly reduce the data and computational resources required.
- **Modular Design:** Building neural networks with interchangeable modules or blocks allows for more flexible architectures that can be adapted to various tasks.
- **Hybrid Models:** Combining different types of neural networks, such as CNNs for feature extraction and RNNs for sequence processing, can harness the strengths of each architecture for complex tasks like video classification or multimodal analysis.

## Implementing a custom model

When constructing a neural network for tasks like character recognition in the EMNIST dataset, it's essential to define key parameters that align with your specific problem. `**input_size**` corresponds to the dimensionality of your input data, `**hidden_size**` is a modifiable parameter that dictates the size of the hidden layers, and `**num_classes**` represents the total number of distinct categories in your classification task.

### In Keras

> [!important]  
> NOTE: Keras documentation for detailed function definitions  

**Model Structure:**  
Begin by initializing a  
`Sequential` model in Keras, then sequentially add layers, starting from input to output. For the EMNIST dataset, a simple model might include a layer to reshape the input, a flattening layer to convert 2D images to 1D vectors, and dense layers for classification purposes:

```Python
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape

input_size = 784  # EMNIST images are 28x28 pixels
hidden_size = 128  # Tunable parameter for the hidden layer
num_classes = 47  # Number of classes in the EMNIST Balanced dataset

model = Sequential([
    Reshape((28, 28, 1), input_shape=(input_size,)),
    Flatten(),
    Dense(hidden_size, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

**Compiling and Training:**  
After defining the model, compile it with the chosen optimizer and loss function, and then train it using the  
`fit` method, specifying your training data, batch size, and epochs:

```Python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=10)
```

### In PyTorch

> [!important]  
> NOTE: PyTorch documentation for detailed function definitions  

**Custom Model Definition:**  
In PyTorch, define a custom neural network by subclassing  
`nn.Module`. Initialize the layers in the constructor and specify the forward pass logic. A straightforward network might include a fully connected layer for the hidden layer and an output layer:

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=47):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**Loss Function and Optimizer Setup:**  
Choose a suitable loss function and optimizer. For classification,  
`CrossEntropyLoss` is typically used, and `Adam` is a widely used optimizer:

```Python
model = SimpleNN(input_size=784, hidden_size=128, num_classes=47)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### **Training the Models**

For training, you'll require a dataset (e.g., EMNIST), a loss function, and an optimizer. Define the number of training epochs and the batch size as well.

**PyTorch Training Loop:**  
In PyTorch, iterate over your dataset in batches, pass the inputs through the model to obtain outputs, compute the loss, and update the model parameters using  
`loss.backward()` and `optimizer.step()`:

```Python
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
```

**Keras Training:**  
In Keras, training the model is straightforward with the  
`fit` method, where `x_train` and `y_train` are your training inputs and labels:

```Python
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

This foundation serves as a starting point for constructing neural networks in both PyTorch and Keras. As you progress to more complex tasks, consider adding more layers, employing different types of layers (like `Conv2D` for image-related tasks), or adjusting parameters and hyperparameters to refine your model's performance.

## Examples with Code

Practical implementations of neural network models provide valuable insights and hands-on experience:

- **CNNs:** Implementations of ResNet and XCeption in Keras for image classification tasks, demonstrating the effectiveness of deep convolutional architectures.
    - [Which animal is this?](https://github.com/christopherseaman/datasci_223/blob/main/exercises/4-classification/practice_1-which_animal.ipynb): A practical exercise in applying CNNs to a multi-class classification problem.
- **Hybrid Models:** Demonstrating the combination of CNNs and RNNs in Keras for tasks that require understanding both spatial and temporal data.
    - [https://github.com/ankangd/HybridCovidLUS](https://github.com/ankangd/HybridCovidLUS): An example of a hybrid model applied to medical imaging for COVID-19 analysis.
- **GPT from Scratch:** Building a simplified version of the GPT model in PyTorch, offering insights into the workings of transformer models.
    - [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT): Accompanied by a [full walkthrough video](https://www.youtube.com/watch?v=kCc8FmEb1nY), this implementation demystifies the GPT architecture.
- **Multi-Model Systems:** Exploring the interaction between different models, such as in GANs, where a generative model is pitted against a discriminative model to produce high-quality synthetic data.
    - [https://github.com/tezansahu/PyTorch-GANs](https://github.com/tezansahu/PyTorch-GANs): A beginner-friendly implementation showing how to train a GAN on the MNIST dataset to generate digit images.

# It came from the internet

## Recent datasci papers from NEJM:

- [**How Censoring Works**](https://evidence.nejm.org/doi/full/10.1056/EVIDstat2300205?emp=marcom&utm_source=nejmglist&utm_medium=email&utm_campaign=evengage23)
- [**Large Language Models**](https://evidence.nejm.org/doi/full/10.1056/EVIDstat2300128?emp=marcom)
- [**The Problem of Multiple Comparisons**](https://evidence.nejm.org/doi/full/10.1056/EVIDstat2200171?emp=marcom&utm_source=nejmglist&utm_medium=email&utm_campaign=evengage23)
- [**Bayesian Way**](https://evidence.nejm.org/doi/full/10.1056/EVIDstat2300090?emp=marcom&utm_source=nejmglist&utm_medium=email&utm_campaign=evengage23)

## More broadly:

https://github.com/TheEconomist/the-economist-war-fire-model

> [!info] Gemma: Introducing new state-of-the-art open models  
> Gemma is a family of lightweight, state-of-the art open models built from the same research and technology used to create the Gemini models.  
> [https://blog.google/technology/developers/gemma-open-models/](https://blog.google/technology/developers/gemma-open-models/)  

> [!info] Video generation models as world simulators  
> We explore large-scale training of generative models on video data.  
> [https://openai.com/research/video-generation-models-as-world-simulators](https://openai.com/research/video-generation-models-as-world-simulators)  

> [!info] Neural network training makes beautiful fractals  
> This blog is intended to be a place to share ideas and results that are too weird, incomplete, or off-topic to turn into an academic paper, but that I think may be important.  
> [https://sohl-dickstein.github.io/2024/02/12/fractal.html](https://sohl-dickstein.github.io/2024/02/12/fractal.html)  

# References

## Preparation for next week (LLMs)

- _What are embeddings?,_ Vicki Boykis [available at the author’s website](https://vickiboykis.com/what_are_embeddings/)

![[LLM_survey.pdf]]

## Books

### Recommendations

- _Python for Data Analysis_, McKinney - author’s [website](https://wesmckinney.com/book/)
- _Python Data Science Handbook,_ VanderPlas - author’s [website](https://jakevdp.github.io/PythonDataScienceHandbook/)
- _PyTorch Tutorials_ - official [documentation](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- _TensorFlow Tutorials_ - official [documentation](https://www.tensorflow.org/tutorials)
- _Dive into Deep Learning -_ authors’ [website](https://d2l.ai)
- _Understanding Deep Learning_ - author’s [website](https://udlbook.github.io/udlbook/) (**WARNING:** intense math)

### [O’Reilly Library Access](https://www.oreilly.com/library-access/) (UCSF institutional access)

- [Hands-on Machine Learning, Géron](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) and companion [repository](https://github.com/ageron/handson-ml3)
- [Machine Learning with PyTorch and Scikit-Learn, Rashka](https://learning.oreilly.com/library/view/machine-learning-with/9781801819312/)
- [Deep Learning with PyTorch, Viehmann](https://learning.oreilly.com/library/view/deep-learning-with/9781617295263/)
- [Machine Learning Design Patterns](https://learning.oreilly.com/library/view/machine-learning-design/9781098115777/), Lakshmanan, et al.

## Tutorials

- **[TensorFlow Tutorials](https://www.tensorflow.org/tutorials)****:** Official tutorials covering various aspects of TensorFlow, from basics to advanced techniques.
- **[PyTorch Tutorials](https://pytorch.org/tutorials/)****:** Collection of tutorials for learning and implementing neural networks using PyTorch.
- **[Keras Documentation](https://keras.io/)****:** Comprehensive guides and tutorials for building neural networks with Keras, a high-level neural networks API.

## Newsletters

- [**Distill**](https://distill.pub/)**:** A journal that offers clear and interactive explanations of machine learning and deep learning concepts.
- [**The Gradient**](https://thegradient.pub/)**:** A publication that focuses on the latest trends and insights in AI and machine learning research.
- **[The Hugging Face Daily Papers](https://huggingface.co/papers)**: A curated list of new research papers from arXiv, each linked to its related models/datasets and Spaces (platform where developers can create, host, and share their ML applications)

## Models

- [https://github.com/microsoft/RespireNet](https://github.com/microsoft/RespireNet) - A CNN-based model designed for COVID-19 severity prediction from lung ultrasound images, showcasing the application of neural networks in healthcare.
- [https://github.com/ritchieng/the-incredible-pytorch](https://github.com/ritchieng/the-incredible-pytorch) - curated list of tutorials, projects, libraries, videos, papers, and books