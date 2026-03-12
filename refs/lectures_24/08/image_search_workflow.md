# Image Search and Download Workflow for Computer Vision Lecture

## Task Definition
Find appropriate images for all FIXME tags in the Computer Vision lecture (@/lectures/08/lecture_08.md ), download them, process them, and update the document.

## Workflow Steps
For each FIXME tag in lectures/08/lecture_08.md:

1. **Search**: Use Google Images via `find_lecture_images.py` to search for a suitable candidate image matching the description
2. **Validate**: Check that downloaded file for validity and appropriateness. If one is a good match, then proceed with processing the image and updating the fixme tag. Otherwise, skip to the next.
3. **Process**: Resize (`magick`) if needed and rename according to the path specified in the FIXME tag
4. **Update**: Modify the FIXME tag in the document to indicate a candidate image has been added
5. **Skip Policy**: If a good image cannot be found quickly, skip and continue to the next FIXME

**NOTE:** Do NOT create placeholder images under any circumstances

## Image Requirements
- Images should be relevant to the described content
- Prefer images that are:
  - Clear and easy to understand
  - Professional/educational in appearance
  - Appropriate resolution (not too small, large images may be scaled down after download)
  - Well-annotated with labels, arrows, or other explanatory elements
  - From reputable sources (academic papers, textbooks, official documentation)
  - Health/medical context when appropriate for the concept

### Image Quality Assessment Criteria
When evaluating candidate images, consider these specific criteria:

1. **Relevance**: How directly does the image illustrate the concept mentioned in the FIXME tag?
2. **Clarity**: Is the image clear and free from unnecessary complexity or distractions?
3. **Annotation**: Does the image include helpful labels, arrows, or other annotations that explain the content?
4. **Context**: Does the image use medical/health examples relevant to the students?
5. **Attribution**: Does the image have proper attribution or come from a reputable source?
6. **Text Readability**: Is any text in the image legible at the intended display size?
7. **Technical Quality**: Is the image properly exposed, in focus, and with good contrast?

### Handling Multiple Good Candidates
When multiple good candidate images are available, prioritize in this order:

1. Images that most directly illustrate the specific concept mentioned in the FIXME tag
2. Images with health/medical context over generic examples
3. Images with clearer annotations and explanations
4. Images from reputable sources (academic papers, textbooks, official documentation)
5. Images with better technical quality (resolution, clarity, contrast)

## Automated Image Search with find_lecture_images.py

The `find_lecture_images.py` script automates the process of finding and downloading candidate images for FIXME tags. It can be run in different modes, including an LLM Agent mode that prepares images for review.

### Basic Usage

```bash
python lectures/08/media/find_lecture_images.py
```

This will:
1. Parse all FIXME tags in the lecture document
2. Search for images based on the tag descriptions
3. Download candidate images for each tag
4. Prompt you to select the best image or skip
5. Copy the selected image to the target path
6. Update the FIXME tag in the document

### LLM Agent Mode

For LLM agents, the script can be run in a special mode that:
1. Creates a structured directory for each FIXME tag
2. Downloads a specified number of candidate images
3. Generates a markdown file with tag information and image previews
4. Organizes everything for easy review and selection

To run in LLM Agent mode:

```bash
python lectures/08/media/find_lecture_images.py --llm-agent --num-results 2
```

This will create directories named `fixme_00`, `fixme_01`, etc. (using 0-based indexing) in `lectures/08/media/downloads/`, each containing:
- Downloaded candidate images
- A `tag_info.md` file with tag details and image previews
- A `tag_info.html` file for better image viewing in a browser

To process a specific tag only:

```bash
python lectures/08/media/find_lecture_images.py --tag-index 0 --llm-agent
```

This processes only the first FIXME tag (index 0).

### Output Format for LLM Agent Mode

The `tag_info.md` file contains:

```markdown
# FIXME Tag Information

## Tag Details
- **Description:** [Tag description]
- **Target Path:** [Target path]
- **Line Number:** [Line number]
- **Original Tag:** `[Original tag text]`

## Candidate Images

### Image 1
- **Path:** [Image filename]
- **Title:** [Image title]
- **URL:** [Image URL]

![Image 1](image_filename)

### Image 2
...
```

Additionally, an HTML version (`tag_info.html`) is generated for better image viewing in a browser. This HTML file provides:

- A more user-friendly interface for reviewing images
- Better image rendering than markdown preview
- Consistent layout for easier comparison between candidates
- Links to original sources

When reviewing images, it's recommended to:
1. Open the HTML file in a browser for better viewing experience
2. Use browser zoom features to examine image details if needed
3. Check image attribution and source information
4. Consider how the image will appear in the final document

### Key Script Options

- `--llm-agent`: Run in LLM agent mode
- `--num-results N`: Download N candidate images per tag (default: 5)
- `--tag-index N`: Process only the tag with index N (0-based)
- `--dry-run`: Don't actually download images or update tags
- `--non-interactive`: Auto-select the first image without prompting

## Manual Technical Process
1. Use browser_action to search Google Images with specific terms from the FIXME description
   ```
   <browser_action>
   <action>launch</action>
   <url>https://www.google.com/search?q=SEARCH_TERMS&tbm=isch</url>
   </browser_action>
   ```

2. Download promising candidates
   ```
   <execute_command>
   <command>curl -o lectures/08/media/downloads/IMAGE_NAME.png IMAGE_URL</command>
   </execute_command>
   ```

3. Check, resize, and rename images
   ```
   <execute_command>
   <command>file lectures/08/media/downloads/IMAGE_NAME.png</command>
   </execute_command>
   
   <execute_command>
   <command>convert lectures/08/media/downloads/IMAGE_NAME.png -resize WIDTHx\> lectures/08/media/TARGET_NAME.png</command>
   </execute_command>
   ```

   Note: The `\>` modifier in the resize command maintains aspect ratio and only resizes if the image is larger than the specified width.

### Image Processing Guidelines

When processing images, follow these guidelines:

- **Maintain aspect ratio**: Always use the `\>` modifier in ImageMagick resize commands to preserve aspect ratio
- **Recommended maximum dimensions**:
  - Logos: 300-400px width
  - Diagrams: 600-800px width
  - Screenshots: 800-1000px width
  - Complex medical images: 1000-1200px width
- **File format preferences**:
  - PNG for diagrams, logos, and screenshots
  - JPG for photographs (with quality setting of 85-90%)
  - GIF only for simple animations
- **Optimization**: Consider file size optimization for web viewing
  ```
  <execute_command>
  <command>convert input.png -strip -quality 85% output.png</command>
  </execute_command>
  ```
- **Consistency**: Maintain consistent styling across similar types of images

4. Update the FIXME tags in the document
   ```
   <search_and_replace>
   <path>lectures/08/lecture_08.md</path>
   <search><!-- #FIXME: Add image: DESCRIPTION lectures/08/media/TARGET_NAME.png --></search>
   <replace><!-- #FIXME: Added candidate image: DESCRIPTION lectures/08/media/TARGET_NAME.png --></replace>
   </search_and_replace>
   ```

## Example Workflow
For a FIXME tag like:
```
<!-- #FIXME: Add image: Pillow logo. lectures/08/media/pillow_logo.png -->
```

1. Search Google Images for "Pillow Python library logo"
   ```
   <browser_action>
   <action>launch</action>
   <url>https://www.google.com/search?q=Pillow+Python+library+logo&tbm=isch</url>
   </browser_action>
   ```

2. Find a suitable image, note its URL, and download it
   ```
   <execute_command>
   <command>curl -o lectures/08/media/downloads/pillow_temp.png IMAGE_URL</command>
   </execute_command>
   ```

3. Check the image and process it
   ```
   <execute_command>
   <command>file lectures/08/media/downloads/pillow_temp.png</command>
   </execute_command>
   
   <execute_command>
   <command>convert lectures/08/media/downloads/pillow_temp.png -resize 300x200 lectures/08/media/pillow_logo.png</command>
   </execute_command>
   ```

4. Update the FIXME tag
   ```
   <search_and_replace>
   <path>lectures/08/lecture_08.md</path>
   <search><!-- #FIXME: Add image: Pillow logo. lectures/08/media/pillow_logo.png --></search>
   <replace><!-- #FIXME: Added candidate image: Pillow logo. lectures/08/media/pillow_logo.png --></replace>
   </search_and_replace>
   ```

## Completion Criteria
- All FIXME tags have either been addressed with appropriate images or explicitly skipped
- The document has been updated to reflect which images have been added
- A list of any skipped images is maintained for future work
- The tracking table is updated with decisions and notes for each tag
- All selected images meet the quality assessment criteria
- Images are properly processed according to the image processing guidelines

## FIXME Tags Tracking
| Status | Decision | Description | Target Path | Notes |
|--------|----------|-------------|-------------|-------|
| [ ] | | Chest X-ray with Nodule Highlighted | lectures/08/media/xray_nodule_example.png | |
| [ ] | | Digital Pathology Slide with Cell Classification | lectures/08/media/pathology_slide_example.png | |
| [ ] | | Robotic Surgery with Computer Vision Assistance | lectures/08/media/robotic_surgery_cv.png | |
| [ ] | | Zoomed-in view of an image showing pixels | lectures/08/media/pixel_grid_example.png | |
| [ ] | | Low Resolution vs. High Resolution | lectures/08/media/resolution_comparison.png | |
| [ ] | | Grayscale X-ray and Pixel Intensity Values | lectures/08/media/grayscale_example.png | |
| [ ] | | RGB Image Decomposed into Red, Green, and Blue Channels | lectures/08/media/rgb_channels_example.png | |
| [ ] | | DICOM Viewer with Image and Metadata | lectures/08/media/dicom_viewer_metadata.png | |
| [ ] | | CT Scan with Different Window/Level Settings | lectures/08/media/dicom_windowing_example.png | |
| [ ] | | Pillow Logo | lectures/08/media/pillow_logo.png | |
| [ ] | | OpenCV Logo | lectures/08/media/opencv_logo.png | |
| [ ] | | Pydicom Conceptual Logo | lectures/08/media/pydicom_logo.png | |
| [ ] | | SimpleITK Logo | lectures/08/media/simpleitk_logo.png | |
| [ ] | | Matplotlib Logo | lectures/08/media/matplotlib_logo.png | |
| [ ] | | Convolutional Filter Operation | lectures/08/media/convolution_filter_static.gif | |
| [ ] | | Valid vs. Same Padding | lectures/08/media/padding_example.png | |
| [ ] | | Input Image and Resulting Feature Maps | lectures/08/media/feature_maps_example.png | |
| [ ] | | ReLU Activation Function | lectures/08/media/relu_function.png | |
| [ ] | | Max Pooling Operation | lectures/08/media/max_pooling_example.png | |
| [ ] | | Typical CNN Architecture Diagram | lectures/08/media/cnn_architecture_diagram.png | |
| [ ] | | Features Learned by Early CNN Layers | lectures/08/media/cnn_early_features.png | |
| [ ] | | Features Learned by Middle CNN Layers | lectures/08/media/cnn_mid_features.png | |
| [ ] | | Features Learned by Deep CNN Layers | lectures/08/media/cnn_deep_features.png | |
| [ ] | | X-ray Classification Example | lectures/08/media/xray_classification_example.png | |
| [ ] | | Dermoscopy Classification Example | lectures/08/media/dermoscopy_classification_example.png | |
| [ ] | | LeNet-5 Architecture | lectures/08/media/lenet5_architecture.png | |
| [ ] | | AlexNet Architecture | lectures/08/media/alexnet_architecture.png | |
| [ ] | | VGG-16 Architecture | lectures/08/media/vgg16_architecture.png | |
| [ ] | | Inception Module | lectures/08/media/inception_module.png | |
| [ ] | | ResNet Residual Block | lectures/08/media/resnet_block.png | |
| [ ] | | DenseNet Connectivity | lectures/08/media/densenet_connectivity.png | |
| [ ] | | Transfer Learning Concept Diagram | lectures/08/media/transfer_learning_diagram.png | |
| [ ] | | Feature Extraction with Transfer Learning | lectures/08/media/feature_extraction_tl.png | |
| [ ] | | Fine-Tuning with Transfer Learning | lectures/08/media/fine_tuning_tl.png | |
| [ ] | | Object Detection Example with Bounding Boxes | lectures/08/media/object_detection_example.png | |
| [ ] | | Anchor Boxes Example | lectures/08/media/anchor_boxes_example.png | |
| [ ] | | Non-Maximum Suppression (NMS) Example | lectures/08/media/nms_example.png | |
| [ ] | | One-Stage vs. Two-Stage Detectors | lectures/08/media/one_vs_two_stage_detectors.png | |
| [ ] | | Intersection over Union (IoU) Diagram | lectures/08/media/iou_diagram.png | |
| [ ] | | Image Segmentation Example: Input and Mask | lectures/08/media/segmentation_example.png | |
| [ ] | | Organ Segmentation in MRI/CT | lectures/08/media/organ_segmentation.png | |
| [ ] | | Tumor Delineation | lectures/08/media/tumor_segmentation.png | |
| [ ] | | Semantic Segmentation Illustration | lectures/08/media/semantic_segmentation_illustration.png | |
| [ ] | | Instance Segmentation Illustration | lectures/08/media/instance_segmentation_illustration.png | |
| [ ] | | U-Net Architecture Diagram | lectures/08/media/unet_architecture_diagram.png | |
| [ ] | | Vision Transformer (ViT) Diagram | lectures/08/media/vit_diagram.png | |
| [ ] | | Generative Model Concept | lectures/08/media/generative_model_concept.png | |
| [ ] | | Grad-CAM Example | lectures/08/media/grad_cam_example.png | |
| [ ] | | Self-Supervised Learning Concept | lectures/08/media/self_supervised_learning.png | |
| [ ] | | Data Augmentation Examples | lectures/08/media/data_augmentation_examples.png | |
| [ ] | | MONAI Logo | lectures/08/media/monai_logo.png | |