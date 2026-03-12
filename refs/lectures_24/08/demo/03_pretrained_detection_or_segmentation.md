# Demo 3: Pre-trained Object Detection or Image Segmentation

**Objective:** Explore the use of pre-trained models for either object detection (identifying and localizing objects) or image segmentation (pixel-level classification). This demo provides two options. The instructor will typically choose one to run live, but students are encouraged to explore both.

**Option A (Recommended for simplicity & common use): Object Detection with YOLO**
*   **Task:** Use a pre-trained YOLO (You Only Look Once) model to detect common objects in an image. YOLO models are known for their speed and accuracy.
*   **Dataset:** A sample image containing common objects (e.g., from the COCO dataset, or a custom image). For a health context, one could try to find a public image with medical equipment if a general YOLO model (trained on COCO) picks them up, or simply use general images to demonstrate the technique.
*   **Tools:** Python, Ultralytics YOLO, OpenCV, Matplotlib, Requests.

**Option B: Image Segmentation with a Pre-trained Model (TensorFlow Hub)**
*   **Task:** Use a pre-trained model (e.g., DeepLabV3) for semantic segmentation, which assigns a class label to every pixel in an image.
*   **Dataset:** A sample image suitable for segmentation (e.g., a general image where salient objects can be segmented).
*   **Tools:** Python, TensorFlow/Keras, TensorFlow Hub, OpenCV, Matplotlib, Requests.


## Option A: Object Detection with YOLO (Ultralytics)

Object detection involves not only classifying objects in an image but also locating them with bounding boxes. YOLO is a popular family of one-stage object detection models. We'll use the `ultralytics` library, which provides an easy-to-use interface for YOLO models.

First, ensure you have `ultralytics` installed. If not, run:
```bash
# pip install ultralytics requests
```

```python
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import requests # For downloading a sample image
from PIL import Image as PILImage # Rename to avoid conflict with potential YOLO internal Image class
from io import BytesIO

# Helper function to display images using Matplotlib
# OpenCV loads images in BGR format, Matplotlib expects RGB.
def show_image_cv(image_bgr, title='Image', figsize=(8,8)):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()
```

### A.1. Load a Pre-trained YOLO Model

We'll load an official YOLOv8 model. `yolov8n.pt` is the "nano" version, which is small and fast, making it good for demos. These models are typically pre-trained on the COCO (Common Objects in Context) dataset, which contains 80 common object categories.

```python
# Load an official YOLOv8 model (e.g., yolov8n.pt for nano, yolov8s.pt for small)
yolo_model_instance = None # Initialize to None
try:
    model_name = 'yolov8n.pt' 
    yolo_model_instance = YOLO(model_name) # This will download the model if not already present
    print(f"Successfully loaded YOLO model: {model_name}")
    print(f"Model classes: {yolo_model_instance.names}") # Display the class names the model can detect
except Exception as e:
    print(f"Error loading YOLO model: {e}")
```
**Example Output:**
```text
Successfully loaded YOLO model: yolov8n.pt
Model classes: {0: 'person', 1: 'bicycle', ..., 79: 'toothbrush'} (List will be long)
```

### A.2. Load a Sample Image

You can use a path to a local image file or download one from the web. We'll use a common example image from Ultralytics.
**Action:** You can change `image_url` to a different image if you like.

```python
# Example: Download a sample image from the web
image_url_yolo = 'https://ultralytics.com/images/bus.jpg' 
# For a medical context, you might try an image of a hospital room or lab,
# but be aware COCO-trained YOLO might not detect specific medical devices unless they resemble common objects.
# image_url_yolo = 'YOUR_MEDICAL_IMAGE_URL_HERE_OR_LOCAL_PATH' 

loaded_image_yolo_rgb = None
try:
    print(f"Attempting to load image from: {image_url_yolo}")
    if image_url_yolo.startswith('http'):
        response = requests.get(image_url_yolo)
        response.raise_for_status() 
        img_pil_yolo = PILImage.open(BytesIO(response.content)).convert('RGB')
    else: # Assume it's a local path
        img_pil_yolo = PILImage.open(image_url_yolo).convert('RGB')
        
    loaded_image_yolo_rgb = np.array(img_pil_yolo) # Convert to NumPy array (RGB)
    print("Image loaded successfully.")
    
    # Display the original image (converting RGB to BGR for our helper function)
    show_image_cv(cv2.cvtColor(loaded_image_yolo_rgb, cv2.COLOR_RGB2BGR), title="Original Sample Image for YOLO")
except Exception as e:
    print(f"Error loading sample image for YOLO: {e}")
```
*(An image window showing the bus (or your chosen image) should appear).*

### A.3. Perform Object Detection

Now, we pass the loaded image (as an RGB NumPy array) to the YOLO model for inference.

```python
if yolo_model_instance is not None and loaded_image_yolo_rgb is not None:
    print("\nPerforming object detection with YOLO...")
    # Perform inference. The model handles the necessary preprocessing.
    yolo_results = yolo_model_instance(loaded_image_yolo_rgb, verbose=False) # verbose=False for less console output
    
    # yolo_results is a list (usually one item for one image).
    # yolo_results[0] contains detections for the first image.
    # Key attributes:
    #   yolo_results[0].boxes : Bounding box coordinates, confidences, classes
    #   yolo_results[0].plot(): Returns an image (NumPy array) with detections plotted.
    
    print(f"Number of detected objects: {len(yolo_results[0].boxes)}")
    if len(yolo_results[0].boxes) > 0:
        print("Example detection details (first box):")
        first_box = yolo_results[0].boxes[0]
        print(f"  Coordinates (xyxy): {first_box.xyxy.cpu().numpy()}")
        print(f"  Confidence: {first_box.conf.cpu().numpy()[0]:.2f}")
        print(f"  Class ID: {int(first_box.cls.cpu().numpy()[0])}")
        print(f"  Class Name: {yolo_model_instance.names[int(first_box.cls.cpu().numpy()[0])]}")

    # --- A.4. Process and Visualize Results ---
    # Ultralytics provides a convenient `.plot()` method to visualize detections.
    try:
        annotated_image_yolo = yolo_results[0].plot(show=False) # Returns a NumPy array (BGR format)
        print("\nDetection results plotted by Ultralytics.")
        show_image_cv(annotated_image_yolo, title="YOLO Object Detection Results")
    except Exception as e:
        print(f"Error during Ultralytics plot: {e}")
else:
    print("Skipping YOLO detection due to model or image loading issues.")
```
**Example Output (for the bus image):**
```text
Performing object detection with YOLO...
Number of detected objects: 5 
Example detection details (first box):
  Coordinates (xyxy): [[108.98  180.92  403.93  443.7 ]]
  Confidence: 0.93
  Class ID: 0
  Class Name: person

Detection results plotted by Ultralytics.
```
*(An image window showing the bus with bounding boxes around detected people and the bus itself, along with class labels and confidence scores, should appear).*

**Expected Outcome (Option A):**
*   The pre-trained YOLOv8n model will be loaded.
*   A sample image will be loaded and displayed.
*   The YOLO model will perform object detection on the image.
*   An image with bounding boxes, class labels, and confidence scores drawn around detected objects will be displayed.


## Option B: Image Segmentation with TensorFlow Hub (e.g., DeepLabV3)

Semantic segmentation classifies each pixel in an image, allowing us to understand not just *what* objects are present, but also their precise shape and location at the pixel level. We'll use a pre-trained DeepLabV3 model from TensorFlow Hub.

Ensure `tensorflow` and `tensorflow_hub` are installed:
```bash
# pip install tensorflow tensorflow_hub requests
```

```python
import tensorflow as tf
import tensorflow_hub as hub
# Other imports (numpy, matplotlib, cv2, PIL, BytesIO, requests) are already done if Option A was run.
# If running standalone, ensure they are imported.

# Helper function to display original image and segmentation mask side-by-side
def show_segmentation(original_rgb_image, segmentation_mask_2d, title='Segmentation', figsize=(12,6)):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # 'viridis' is a good colormap for segmentation masks.
    # The mask contains class indices for each pixel.
    plt.imshow(segmentation_mask_2d, cmap='viridis') 
    plt.title('Segmentation Mask')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- B.1. Load a Pre-trained Segmentation Model using KaggleHub ---
# We'll use a DeepLabV3 model with a MobileNetV2 backbone, pre-trained on PASCAL VOC.
# This model expects 224x224 RGB images.
# Instead of using TensorFlow Hub directly, we'll use kagglehub which is more reliable

# First, ensure you have kagglehub installed
# pip install kagglehub

import kagglehub
import tensorflow as tf

# Download the DeepLabV3 model
model_path = None
tf_hub_segmentation_model = None

try:
    # Download latest version of the model
    print("Downloading DeepLabV3 model from KaggleHub...")
    model_path = kagglehub.model_download("tensorflow/deeplabv3/tfLite/default")
    print(f"Model downloaded to: {model_path}")
    
    # List files in the model directory to find the correct model file
    import os
    print("Files in the model directory:")
    for root, dirs, files in os.walk(model_path):
        for file in files:
            print(f"  {os.path.join(root, file)}")
    
    # Check if this is a SavedModel (has saved_model.pb file)
    is_saved_model = os.path.exists(os.path.join(model_path, "saved_model.pb"))
    
    # Create a wrapper function to make predictions
    if is_saved_model:
        print("Loading as a SavedModel...")
        # Load as a SavedModel
        loaded_model = tf.saved_model.load(model_path)
        model_func = loaded_model.signatures["serving_default"]
        
        # Create a wrapper function
        def segmentation_model(input_tensor):
            # Get the input tensor name from the model signature
            input_name = list(model_func.structured_input_signature[1].keys())[0]
            result = model_func(**{input_name: input_tensor})
            # Get the output tensor (usually the first or only output)
            output_name = list(result.keys())[0]
            return result[output_name]
            
        print("SavedModel loaded successfully.")
    else:
        # Look for .tflite files
        tflite_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith('.tflite'):
                    tflite_files.append(os.path.join(root, file))
        
        if tflite_files:
            print(f"Found TFLite model file(s): {tflite_files}")
            model_file = tflite_files[0]  # Use the first .tflite file found
            
            # Load as TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_file)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Create a wrapper function
            def segmentation_model(input_tensor):
                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()
                return interpreter.get_tensor(output_details[0]['index'])
                
            print("TensorFlow Lite model loaded successfully.")
        else:
            # Try using a different model from TensorFlow Hub directly
            print("No TFLite files found and not a SavedModel. Trying TensorFlow Hub...")
            tf_hub_url = "https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1"
            tf_hub_segmentation_model = hub.load(tf_hub_url)
            
            # Create a wrapper function
            def segmentation_model(input_tensor):
                return tf_hub_segmentation_model(input_tensor)
                
            print("TensorFlow Hub model loaded successfully.")
    
    print(f"Model input shape: {input_details[0]['shape']}")
    print(f"Model output shape: {output_details[0]['shape']}")
    
    # Set the model function
    tf_hub_segmentation_model = segmentation_model
    print("Segmentation model loaded successfully.")
except Exception as e:
    print(f"Error loading segmentation model: {e}")

# --- B.2. Load and Preprocess a Sample Image for Segmentation ---
# The model expects input images of size 257x257, normalized to [0,1].
# This is based on the model's input shape: [1, 257, 257, 3]
IMG_HEIGHT_SEG = 257
IMG_WIDTH_SEG = 257

# You can use a different image for segmentation.
image_url_segmentation = 'https://images.pexels.com/photos/617278/pexels-photo-617278.jpeg?auto=compress&cs=tinysrgb&w=600' # A cat image

loaded_image_seg_original_rgb = None
processed_image_seg_batch = None

try:
    print(f"Attempting to load image for segmentation from: {image_url_segmentation}")
    if image_url_segmentation.startswith('http'):
        response_seg = requests.get(image_url_segmentation)
        response_seg.raise_for_status()
        img_pil_seg = PILImage.open(BytesIO(response_seg.content)).convert('RGB')
    else: # Assume local path
        img_pil_seg = PILImage.open(image_url_segmentation).convert('RGB')
        
    loaded_image_seg_original_rgb = np.array(img_pil_seg) # Keep original for display

    # Preprocess: Resize to model's expected input size and Normalize to [0,1]
    img_resized_pil_seg = img_pil_seg.resize((IMG_WIDTH_SEG, IMG_HEIGHT_SEG))
    img_resized_np_seg = np.array(img_resized_pil_seg)
    
    processed_image_seg_normalized = img_resized_np_seg.astype(np.float32) / 255.0
    # Add batch dimension as the model expects a batch of images
    processed_image_seg_batch = np.expand_dims(processed_image_seg_normalized, axis=0) 

    print(f"Image for segmentation loaded and preprocessed. Batch shape: {processed_image_seg_batch.shape}")
    
    # Display the (resized) original image that will be fed to the model
    plt.figure(figsize=(6,6))
    plt.imshow(img_resized_np_seg) # Show the resized version that matches model input
    plt.title("Preprocessed Image for Segmentation Model")
    plt.axis("off")
    plt.show()
    
except Exception as e:
    print(f"Error loading or preprocessing image for segmentation: {e}")

# --- B.3. Perform Image Segmentation ---
if tf_hub_segmentation_model is not None and processed_image_seg_batch is not None:
    print("\nPerforming image segmentation with loaded model...")
    try:
        # The model outputs logits (raw scores) for each class at each pixel.
        # Output shape is typically (batch_size, height, width, num_classes).
        # For this model: (1, 257, 257, 21)
        segmentation_logits = tf_hub_segmentation_model(processed_image_seg_batch)
        
        # To get the final segmentation mask, we take the argmax along the class dimension.
        # This gives the class index with the highest score for each pixel.
        segmentation_mask_2d = tf.argmax(segmentation_logits, axis=-1)
        segmentation_mask_2d = segmentation_mask_2d[0].numpy() # Remove batch dim and convert to NumPy array
        
        print(f"Segmentation mask generated. Shape: {segmentation_mask_2d.shape}") # Should be (257, 257)

        # --- B.4. Visualize Segmentation Result ---
        # Display the original (resized for fair comparison) and the segmentation mask
        show_segmentation(img_resized_np_seg, segmentation_mask_2d)

    except Exception as e:
        print(f"Error during segmentation inference or visualization: {e}")
else:
    print("Skipping segmentation due to model or image loading/processing issues.")

```
**Example Output (for the cat image):**
```text
Downloading DeepLabV3 model from KaggleHub...
Model downloaded to: /Users/username/.cache/kagglehub/models/tensorflow/deeplabv3/tfLite/default/1
Files in the model directory:
  /Users/username/.cache/kagglehub/models/tensorflow/deeplabv3/tfLite/default/1/saved_model.pb
  /Users/username/.cache/kagglehub/models/tensorflow/deeplabv3/tfLite/default/1/variables/variables.index
  /Users/username/.cache/kagglehub/models/tensorflow/deeplabv3/tfLite/default/1/variables/variables.data-00000-of-00001
  /Users/username/.cache/kagglehub/models/tensorflow/deeplabv3/tfLite/default/1/assets/labels.txt
Loading as a SavedModel...
SavedModel loaded successfully.
Segmentation model loaded successfully.
Attempting to load image for segmentation from: https://images.pexels.com/photos/617278/pexels-photo-617278.jpeg?auto=compress&cs=tinysrgb&w=600
Image for segmentation loaded and preprocessed. Batch shape: (1, 257, 257, 3)

Performing image segmentation with loaded model...
Segmentation mask generated. Shape: (257, 257)
```
*(Two images should be displayed side-by-side: the preprocessed cat image, and its segmentation mask where different colors represent different PASCAL VOC classes like 'cat', 'background', etc.).*

**Expected Outcome (Option B):**
*   A pre-trained segmentation model (e.g., DeepLabV3) will be loaded from TensorFlow Hub.
*   A sample image will be loaded, preprocessed (resized, normalized), and displayed.
*   The model will perform semantic segmentation on the image.
*   The original image (resized) and the resulting segmentation mask will be displayed side-by-side. The mask will show different regions colored according to the class predicted for each pixel.
