# Lecture 09 — Computer Vision: Speaking Notes

## What Is Computer Vision?

- Open with the breadth of CV: it's not just classification — detection, segmentation, tracking, generation, reconstruction
- Healthcare angle: radiology, pathology, dermatology, ophthalmology, surgery — CV is already deployed in FDA-approved devices
- When discussing pixels: students may not realize that "images are just numbers" — emphasize this concretely with the pixel grid visual
- DICOM: mention that every hospital system uses DICOM, and students will encounter it in any medical imaging project. The metadata is as important as the pixels
- Common misconception: students think they need massive datasets. Transfer learning section will address this

## CNNs in PyTorch

- Students know CNN concepts from Keras (lecture 06). Don't re-explain convolutions from scratch — focus on the PyTorch API differences
- Key point: PyTorch uses NCHW (channels first), Keras uses NHWC (channels last). This causes shape errors constantly
- nn.Sequential is the closest to Keras Sequential — show it as the "familiar" pattern
- The training loop: this is the biggest difference from Keras. Walk through each line. Students are used to model.fit() doing everything
- Emphasize: the explicit loop is more work but gives you complete control — you can add custom logging, learning rate schedules, gradient clipping, etc.
- Common mistake: forgetting optimizer.zero_grad() — gradients accumulate by default in PyTorch

## torchvision — The Computer Vision Toolkit

- Transforms: emphasize the train vs eval distinction. Students often apply augmentation at test time too
- ImageNet normalization: explain why we use these specific numbers — the pretrained model saw data normalized this way, so we need to match
- Augmentation philosophy: each augmented image is a "new" training example the model hasn't seen before. More diversity → better generalization
- DataLoader: num_workers > 0 can cause issues on some systems (especially Windows/Mac). Set to 0 for debugging
- Common mistake: not using separate transform pipelines for train and eval

## Transfer Learning & Pretrained Models

- This is the most important practical section. Most real CV work uses transfer learning
- Analogy: it's like learning a new language when you already know one — you don't start from zero
- Feature extraction vs fine-tuning: start with feature extraction (faster, works with less data), try fine-tuning if you need better performance
- The weights parameter: show the modern API (weights="DEFAULT") vs the old way (pretrained=True)
- timm: mention it's the go-to for researchers and practitioners who need specific architectures
- Practical tip: ResNet-18 is a great starting point — small, fast, works well on most tasks

## Evaluating Vision Models

- Remind students of the class imbalance problem from lecture 05 — it's even worse in medical imaging
- A 98% accuracy model that misses 50% of cancers is dangerous — always look at per-class metrics
- Confusion matrix: students should be able to read one and identify which classes are confused
- For detection/segmentation: IoU and Dice are new — explain the geometric intuition

## Object Detection

- This section is conceptual + practical usage of pretrained models
- Students don't need to implement a detector from scratch — they need to know how to use one
- YOLO vs Faster R-CNN: YOLO is faster (real-time), Faster R-CNN is more accurate. Both work
- For medical use: the pretrained models (COCO) won't detect medical findings out of the box — you need to fine-tune
- NMS: walk through the visual. Students find this counterintuitive at first

## Image Segmentation

- U-Net is the workhorse of medical segmentation — students should know its architecture conceptually
- Skip connections are the key insight: they pass spatial detail from encoder to decoder
- Loss functions: Dice loss is the standard for medical segmentation because it handles class imbalance
- smp (segmentation_models_pytorch): emphasize that you get U-Net with any pretrained backbone in one line
- MONAI: mention for students interested in research — it's the PyTorch framework for medical imaging

## Advanced Topics

- This is a survey — awareness, not mastery
- ViTs: mention that they're increasingly replacing CNNs, especially for large-scale tasks
- SAM: demonstrate if time permits — it's impressive and students will encounter it
- Explainability: critical for clinical adoption. Grad-CAM is the most accessible method
- Self-supervised: mention the annotation bottleneck in medical imaging — this is how the field is addressing it

## Demo Pacing

- Demo 1 (DICOM + transforms): ~15 min. Focus on the DICOM loading and transform visualization
- Demo 2 (transfer learning): ~20 min. This is the core demo — make sure training loop is clear
- Demo 3 (detection + segmentation): ~15 min. Show pretrained inference, discuss medical applications

## Common Student Issues

- CUDA out of memory: reduce batch size or use CPU
- Shape mismatch errors: usually NCHW vs NHWC confusion
- Model not learning: check that only the head is unfrozen, learning rate is appropriate
- DataLoader hanging: set num_workers=0 on Windows/Mac
- Import errors: make sure torchvision version matches torch version
