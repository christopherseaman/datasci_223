# Demo 1: Image Loading, Preprocessing & Basic CNN Structure

**Objective:** Learn to load, inspect, and preprocess medical images (DICOM and standard formats). Optionally, visualize the structure of a basic CNN.

**Dataset:**
*   A sample DICOM image (000001.dcm) located in the images directory
*   A sample medical chest X-ray image (medical_DICOM_chest_xray_1.jpg) located in the images directory

**Tools:** Python, Pydicom, Pillow, OpenCV, Matplotlib, TensorFlow/Keras.


## 1. Setup and Imports

First, we import all necessary libraries.
*   `pydicom` is for reading and manipulating DICOM files.
*   `PIL (Pillow)` and `cv2 (OpenCV)` are for general image processing tasks.
*   `numpy` is for numerical operations, especially with image arrays.
*   `matplotlib.pyplot` is for displaying images.
*   `tensorflow` and specific Keras layers are for building the neural network structure.

```python
import pydicom
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Helper function to display images using Matplotlib
def show_image(image, title='Image', cmap=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()
```


## 2. Loading and Inspecting a DICOM Image

DICOM (Digital Imaging and Communications in Medicine) is a standard format for storing and transmitting medical images. It contains not only pixel data but also rich metadata about the patient, study, and imaging equipment. We use the `pydicom` library to work with these files.

**Note:** We'll use a sample DICOM file located in the images directory.

```python
# Path to the DICOM file
dicom_path = 'images/000001.dcm'

try:
    # Load DICOM file
    ds = pydicom.dcmread(dicom_path)

    # The 'ds' object now holds the DICOM dataset, including metadata and pixel data.
    # Let's print some common metadata fields.
    print("DICOM Metadata:")
    print(f"Patient's Name: {ds.PatientName}")
    if hasattr(ds, 'StudyDescription'):
        print(f"Study Description: {ds.StudyDescription}")
    else:
        print("Study Description: Not available")
    print(f"Modality: {ds.Modality}")
    print(f"Image Shape: {ds.pixel_array.shape}") # Access pixel data via pixel_array
    # PixelSpacing might not be present in all DICOMs, or might be multi-valued
    if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing:
        print(f"Pixel Spacing: {ds.PixelSpacing}")
    else:
        print("Pixel Spacing: Not available or empty")
    
    # Get pixel data
    dicom_image = ds.pixel_array
    show_image(dicom_image, title=f'DICOM Image ({ds.Modality})', cmap='gray')

    # Basic pixel value inspection
    print(f"\nDICOM Image pixel stats:")
    print(f"Min pixel value: {dicom_image.min()}")
    print(f"Max pixel value: {dicom_image.max()}")
    print(f"Data type: {dicom_image.dtype}")


except FileNotFoundError:
    print(f"Error: DICOM file not found at {dicom_path}. Please provide a valid path.")
except Exception as e:
    print(f"An error occurred while processing the DICOM file: {e}")
```

**Example Output (will vary based on your DICOM file):**
```text
DICOM Metadata:
Patient's Name: ANONYMIZED
Study Description: CHEST (PA AND LAT)
Modality: CR
Image Shape: (1024, 1024)
Pixel Spacing: [0.168, 0.168]

DICOM Image pixel stats:
Min pixel value: 0
Max pixel value: 4095
Data type: uint16
```
*(An image window showing the DICOM content should also appear).*

**Expected Outcome:** The DICOM image should be displayed, and selected metadata (like PatientName, StudyDescription, Modality, image dimensions, pixel spacing) should be printed. Pixel statistics will also be shown.

## Print DICOM metadata using pydicom

```python
ds = pydicom.dcmread('images/000001.dcm')
print(ds)
```

## 3. Loading and Inspecting a Standard Image (PNG/JPEG)

For non-DICOM images like PNG or JPEG, we can use libraries like Pillow (PIL) or OpenCV. These are common in general computer vision and also useful when dealing with medical images that have been converted to these formats.

**Note:** We'll use a sample medical image located in the images directory.

```python
# Path to the standard image file
standard_image_path = 'images/medical_DICOM_chest_xray_1.jpg'

try:
    # Load with Pillow
    pil_image = Image.open(standard_image_path)
    print("\nPillow Image Info:")
    print(f"Format: {pil_image.format}")
    print(f"Size: {pil_image.size}") # (width, height)
    print(f"Mode: {pil_image.mode}") # e.g., 'RGB', 'L' (grayscale)
    # pil_image.show() # This would open the image in your system's default image viewer

    # Convert Pillow image to NumPy array for Matplotlib display and further processing
    pil_image_np = np.array(pil_image)
    show_image(pil_image_np, title='Standard Image (Loaded with Pillow)')

    # Load with OpenCV
    # OpenCV reads images in BGR format by default.
    cv_image = cv2.imread(standard_image_path) 
    if cv_image is not None:
        print("\nOpenCV Image Info:")
        print(f"Shape: {cv_image.shape}") # (height, width, channels)
        print(f"Data type: {cv_image.dtype}")

        # Convert BGR to RGB for Matplotlib display
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        show_image(cv_image_rgb, title='Standard Image (Loaded with OpenCV - Displayed as RGB)')
    else:
        print(f"Error: Could not load image with OpenCV from {standard_image_path}")

except FileNotFoundError:
    print(f"Error: Standard image file not found at {standard_image_path}. Please provide a valid path.")
except Exception as e:
    print(f"An error occurred while processing the standard image file: {e}")
```

**Example Output (for a 224x224 RGB PNG image):**
```text
Pillow Image Info:
Format: PNG
Size: (224, 224)
Mode: RGB

OpenCV Image Info:
Shape: (224, 224, 3)
Data type: uint8
```
*(Two image windows should appear, one from Pillow and one from OpenCV).*

**Expected Outcome:** The standard image (PNG/JPEG) should be displayed using both Pillow (converted to NumPy) and OpenCV. Information like format, size, mode (Pillow), and shape (OpenCV) will be printed.


## 4. Basic Image Preprocessing

Preprocessing is a crucial step in any computer vision pipeline. Common operations include resizing, normalization, and color conversion.

### 4.1. Resizing

Images often come in various sizes. For many machine learning models, especially CNNs, a fixed input size is required. Resizing standardizes these dimensions. We'll use a target size of 128x128 pixels as an example.

```python
if 'cv_image_rgb' in locals() and cv_image_rgb is not None:
    # Resizing with OpenCV
    # OpenCV's resize function takes (width, height) for the new size.
    new_size_cv_1 = (64, 64) # (width, height) for OpenCV
    resized_image_cv_1 = cv2.resize(cv_image_rgb, new_size_cv_1, interpolation=cv2.INTER_AREA)
    show_image(resized_image_cv_1, title=f'Resized Image (OpenCV) to {new_size_cv_1}')

    new_size_cv_2 = (128, 128) # (width, height) for OpenCV
    resized_image_cv_2 = cv2.resize(cv_image_rgb, new_size_cv_2, interpolation=cv2.INTER_AREA)
    show_image(resized_image_cv_2, title=f'Resized Image (OpenCV) to {new_size_cv_2}')

    # Resizing with Pillow
    # Pillow's resize method also takes (width, height).
    pil_image_to_resize = Image.fromarray(cv_image_rgb) # Create a PIL image from the OpenCV RGB version
    new_size_pil_1 = (64, 64) # (width, height) for Pillow
    resized_image_pil_1 = pil_image_to_resize.resize(new_size_pil_1)
    show_image(np.array(resized_image_pil_1), title=f'Resized Image (Pillow) to {new_size_pil_1}')

    new_size_pil_2 = (128, 128) # (width, height) for Pillow
    resized_image_pil_2 = pil_image_to_resize.resize(new_size_pil_2)
    show_image(np.array(resized_image_pil_2), title=f'Resized Image (Pillow) to {new_size_pil_2}')
else:
    print("Skipping resizing as 'cv_image_rgb' is not available (likely previous image load failed).")
```
**Expected Outcome:** The image will be resized to 128x128 pixels and displayed, once using OpenCV and once using Pillow. Both should look identical.

### 4.2. Normalization (Pixel Values to 0-1 Range)

Normalization scales pixel values to a standard range, typically [0, 1] or [-1, 1]. This helps in stabilizing and speeding up the training of neural networks. For images with pixel values from 0 to 255, a common way to normalize to [0, 1] is by dividing by 255.

```python
if 'resized_image_cv' in locals() and resized_image_cv is not None:
    # Ensure the image is float type before division for normalization
    normalized_image = resized_image_cv.astype(np.float32) / 255.0
    
    print("\nNormalized Image Stats:")
    print(f"Min pixel value: {normalized_image.min()}")
    print(f"Max pixel value: {normalized_image.max()}")
    print(f"Data type: {normalized_image.dtype}")
    
    # Displaying a float image might look different if not handled correctly by imshow,
    # but Matplotlib handles [0,1] float images well.
    show_image(normalized_image, title='Normalized Image (0-1 Range)')
else:
    print("Skipping normalization as 'resized_image_cv' is not available.")
```
**Example Output:**
```text
Normalized Image Stats:
Min pixel value: 0.0
Max pixel value: 1.0
Data type: float32
```
*(An image window showing the normalized image should appear. It might look identical to the resized image if the original was 8-bit, as Matplotlib scales display by default, but the underlying data values are now between 0 and 1).*

**Expected Outcome:** Pixel value statistics (min/max) for the normalized image will be printed, showing they are within the 0-1 range. The image will be displayed.

### 4.3. Color Conversion (e.g., RGB to Grayscale)

Sometimes, color information is not necessary or can even be a distraction for a model. Converting an image to grayscale reduces the number of channels (from 3 for RGB to 1 for grayscale), simplifying the input.

```python
if "cv_image_rgb" in locals() and cv_image_rgb is not None:
    # Load a colorful image (e.g., puppy.jpg)
    color_image_path = "images/puppy.jpg"
    color_image = cv2.imread(color_image_path)
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    show_image(color_image_rgb, title="Colorful Image")

    # Convert RGB to Grayscale using OpenCV
    grayscale_image_cv = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2GRAY)
    show_image(
        grayscale_image_cv, title="Grayscale Image (OpenCV)", cmap="gray"
    )
    print(f"\nOpenCV Grayscale Image Shape: {grayscale_image_cv.shape}")

    # Split the color image into channels
    b, g, r = cv2.split(color_image_rgb)

    # Display the channels
    show_image(r, title="Red Channel", cmap="gray")
    show_image(g, title="Green Channel", cmap="gray")
    show_image(b, title="Blue Channel", cmap="gray")

    # Convert RGB to Grayscale using Pillow
    # First, ensure we have a PIL Image object of the resized color image
    pil_image_for_gray_conversion = Image.fromarray(color_image_rgb)
    grayscale_image_pil = pil_image_for_gray_conversion.convert(
        "L"
    )  # 'L' mode is for grayscale
    print(
        f"Pillow Grayscale Image Shape: {np.array(grayscale_image_pil).shape}"
    )

else:
    print("Skipping color conversion as 'resized_image_cv' is not available.")

```
**Example Output:**
```text
OpenCV Grayscale Image Shape: (128, 128)
Pillow Grayscale Image Shape: (128, 128)
```
*(Two image windows showing the grayscale converted image should appear, one from OpenCV and one from Pillow. They should look identical).*

**Expected Outcome:** The resized color image will be converted to grayscale and displayed, once using OpenCV and once using Pillow. Their shapes will be printed, showing a single channel.


## 5. (Optional) Build and Visualize a Simple CNN Structure

This section demonstrates how to define a simple Convolutional Neural Network (CNN) using TensorFlow/Keras. We will not train this model here, only define its architecture and look at its summary.

CNNs typically expect input images with a specific shape: `(height, width, channels)`.
For example, if we use our resized RGB images (128x128 pixels, 3 channels), the `input_shape` would be `(128, 128, 3)`.
If we were to use the grayscale images (128x128 pixels, 1 channel), it would be `(128, 128, 1)`.

```python
# Define the input shape based on our preprocessed images (e.g., 128x128 RGB)
input_shape_example = (128, 128, 3) 
# Or for grayscale:
# input_shape_example = (128, 128, 1) 

# Define a simple CNN model
simple_cnn_model = Sequential([
    # First Convolutional Layer: 32 filters, 3x3 kernel, ReLU activation.
    # 'padding="same"' ensures the output feature map has the same height/width as the input (for this layer, considering stride=1).
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_example, padding='same'),
    # MaxPooling Layer: Reduces dimensionality by taking the max value in 2x2 windows.
    MaxPooling2D((2, 2)),
    
    # Second Convolutional Layer: 64 filters.
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    # Third Convolutional Layer: 128 filters.
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    # Flatten Layer: Converts the 3D feature maps into a 1D vector for the Dense layers.
    Flatten(),
    
    # Dense Layer: A fully connected layer with 128 units and ReLU activation.
    Dense(128, activation='relu'),
    
    # Output Layer: Dense layer with 'num_classes' units (e.g., 10 for 10 classes)
    # 'softmax' activation is used for multi-class classification to output probabilities.
    Dense(10, activation='softmax') 
])

# Print model summary to see the architecture and number of parameters.
print("\nSimple CNN Model Summary:")
simple_cnn_model.summary()

# (Optional) Visualize the model structure if graphviz and pydot are installed
# This can be very helpful for understanding the flow of data through the network.
# You might need to install them: pip install graphviz pydot pydotplus
# And ensure Graphviz executables are in your system's PATH.
# try:
#     tf.keras.utils.plot_model(
#         simple_cnn_model, 
#         to_file='simple_cnn_model.png', 
#         show_shapes=True, 
#         show_dtype=False, # Optionally show data types
#         show_layer_names=True,
#         rankdir='TB', # 'TB' for top-to-bottom, 'LR' for left-to-right
#         expand_nested=False, # Expand nested models
#         dpi=96 # Dots per inch
#     )
#     print("\nModel structure visualization saved to simple_cnn_model.png")
#     # If in a Jupyter Notebook, you can display it directly:
#     # from IPython.display import Image as IPImage
#     # display(IPImage('simple_cnn_model.png'))
# except ImportError:
#     print("\nCould not import pydot or graphviz. Skipping model plot generation. Ensure pydot and graphviz are installed.")
# except Exception as e:
#     print(f"\nCould not plot model due to an error: {e}")

```

**Example Model Summary Output:**
```text
Simple CNN Model Summary:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 128, 128, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2  (None, 64, 64, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 64, 64, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooli  (None, 32, 32, 64)        0         
 ng2D)                                                           
                                                                 
 conv2d_2 (Conv2D)           (None, 32, 32, 128)       73856     
                                                                 
 max_pooling2d_2 (MaxPooli  (None, 16, 16, 128)       0         
 ng2D)                                                           
                                                                 
 flatten (Flatten)           (None, 32768)             0         
                                                                 
 dense (Dense)               (None, 128)               4194432   
                                                                 
 dense_1 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 4,288,970
Trainable params: 4,288,970
Non-trainable params: 0
_________________________________________________________________
```

**Expected Outcome:** The summary of the simple CNN model (layers, output shapes, number of parameters) will be printed. If `graphviz` and `pydot` are installed and correctly configured, a diagram of the model architecture might be generated and saved as `simple_cnn_model.png`.


**Self-Check / Validation:**
*   Were you able to load both DICOM and standard image files without errors (assuming valid paths)?
*   Did the printed metadata and image properties match your expectations for the sample images?
*   Were the images displayed correctly at each step (original, resized, normalized, grayscale)?
*   If you ran the optional CNN part, did the model summary print correctly? Did the model visualization get created (if dependencies were met)?