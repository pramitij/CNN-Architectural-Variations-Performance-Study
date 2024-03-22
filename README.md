# CNN-Architectural-Variations-Performance-Study
>In this project, I investigate how various architectural modifications within a Convolutional Neural Network (CNN) impact its ability to classify CIFAR-10 images. Specifically, I explore the effects of Batch Normalization, Dropout, the number of convolution and pooling layers, and different activation functions on the model's performance.

## Experiment Details and Findings
### Batch Normalization
I evaluated the model performance across different batch sizes to find the optimal setting for accuracy and loss ratio. The experiment revealed that a batch size of 64 achieves higher accuracy, while a batch size of 128 offers a better balance between accuracy and loss.


### Dropout
By varying dropout rates, I discovered that a dropout rate of 0.4 provides the best compromise between accuracy and loss for the model when evaluating testing data.


### Convolution and Pooling Layers
The experiment on varying the number of convolution and pooling layers indicated that models with more layers show improved generalization capabilities, particularly noted by the decreased discrepancy between training and testing accuracy.


### Activation Function
Among the tested activation functions (ReLU, Tanh, Sigmoid, Swish), ReLU demonstrated the best performance in terms of achieving the highest accuracy and lowest loss on test data.


## Technical Implementation
### Dependencies
TensorFlow: For building and training the CNN models.
Keras: TensorFlow's high-level API for easy model construction.
NumPy: For numerical operations and data manipulation.
Matplotlib: For plotting and visualizing the data and results.

### File Structure
BatchNormalization.ipynb: Analyzes the impact of batch normalization on model performance.
DropOutRates.ipynb: Examines the effects of varying dropout rates.
ConvolutionPoolingLayers.ipynb: Investigates how different numbers of convolution and pooling layers affect the model.
ActivationFunction.ipynb: Compares the performance of various activation functions.

## Usage Instructions
Ensure all dependencies are installed in your Python environment.
Clone the repository and navigate to the project directory.
Open and execute each notebook to replicate the experiments or to conduct your analysis.
Visualize the results within the notebooks for comprehensive insights into the model's performance under different architectural modifications.

## Conclusion
This detailed exploration provides valuable insights into how specific architectural choices can significantly influence the performance of CNN models. These findings underscore the importance of thoughtful experimentation in the design of deep learning models for image classification tasks.

