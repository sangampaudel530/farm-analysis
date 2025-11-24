# Farm Land Analysis Team - Segmentation and Object Detection Project

## Project Overview

The Farm Land Analysis Team is responsible for developing an integrated computer vision system that combines semantic segmentation and object detection to analyze agricultural landscapes from high-resolution orthomosaic imagery for agricultural monitoring and post-disaster assessment of farming regions across Nepal's valleys.

## Objective

Develop a dual-component model capable of:

1. **Semantic Segmentation**: Pixel-wise classification of land cover types
2. **Object Detection**: Identification and localization of discrete agricultural features

### Target Classes

- **Agro Lands/Farms**: Cultivated agricultural fields (rice paddies, crop fields)
- **Trees Canopy**: Forest cover and tree clusters
- **Sand Deposition**: Areas affected by riverine sand deposition (land degradation indicator)
- **Land Usage**: Additional land cover categories (fallow land, grassland, mixed-use areas)

The model must provide both continuous segmentation masks for land cover analysis and bounding boxes with class labels for discrete object detection.

## Technical Context

### Understanding Orthomosaic Data

Please use this georeferenced TIFF orthomosaics generated from drone imagery.

**Full orthomosaics (Google Drive):**

* Pre-flood: **[[Drive link – pre]](https://drive.google.com/file/d/1by9yXKye9QkaN9dAqWnMJkNglG9AD3rM/view?usp=sharing)**
* Post-flood: **[[Drive link – post]](https://drive.google.com/file/d/1x8VdZBs25F9EkWnJw1XR4xnUfBHn0eyY/view?usp=share_link)**

### Dual-Model Architecture

The project requires two complementary outputs:

#### Semantic Segmentation Component
Pixel-wise classification producing continuous masks across the entire orthomosaic. Recommended architectures:

- **U-Net**: Standard baseline for semantic segmentation
- **DeepLabV3+**: Atrous convolutions for multi-scale feature extraction
- **SegFormer**: Transformer-based architecture with strong performance on aerial imagery
- **HRNet**: High-resolution representation maintenance throughout the network
- **UPerNet**: Unified perceptual parsing for complex land cover scenes

#### Object Detection Component
Bounding box detection for discrete agricultural features. Recommended architectures:

- **Faster R-CNN**: Two-stage detector with region proposals
- **YOLO (v8/v9/v11)**: Single-stage real-time detection
- **RetinaNet**: Focal loss for handling class imbalance
- **DINO**: End-to-end transformer detector with strong small object performance
- **Co-DETR**: Enhanced DETR variant with collaborative detection

## Annotation Strategy

The team must annotate data for both segmentation and detection tasks:

### Annotation Workflow Options

**Option 1: Tile-Based Approach**
- Divide orthomosaic into manageable tiles (1024x1024 or 2048x2048 pixels)
- Annotate each tile for both segmentation and detection
- Advantages: Compatible with standard tools, manageable file sizes
- Considerations: Handle boundary cases where features span multiple tiles

**Option 2: Full Orthomosaic Annotation**
- Annotate directly on full-resolution imagery
- Requires QGIS with annotation plugins or specialized geospatial annotation platforms
- Advantages: Maintains geographic context, reduces boundary artifacts
- Considerations: Resource-intensive, requires robust infrastructure

### Recommended Annotation Tools
- **Label Studio**: Supports both segmentation and detection annotations
- **CVAT**: Comprehensive annotation capabilities with polygon and bbox support
- **Roboflow**: Handles tiling, annotation, and format conversion
- **QGIS + Plugins**: Direct annotation on georeferenced imagery with GIS integration

## Deliverables

### 1. Annotated Agricultural Dataset
- Complete annotations for all four land cover classes
- Separate annotation files for segmentation and detection tasks
- Dataset split into training (70%), validation (15%), and test (15%) sets
- Metadata documentation:
  - Total instances and area coverage per class
  - Annotation methodology and guidelines
  - Pixel resolution and coordinate reference system
  - Class distribution statistics

### 2. Trained Models

**Segmentation Model:**
- Model weights and configuration files
- Training logs with loss curves and metrics (IoU, F1-score per class)
- Inference script for generating segmentation masks

**Detection Model:**
- Model weights and configuration files
- Training logs with detection metrics (mAP, precision, recall per class)
- Inference script for generating bounding boxes

### 3. Evaluation Report

Comprehensive document containing:

- **Quantitative Metrics**: Segmentation (Mean IoU, per-class IoU, F1-score) and Detection (mAP@0.5, mAP@0.75, precision, recall per class)
- **Qualitative Analysis**: Visual examples of model predictions overlaid on test images and failure case analysis
- **Methodology**: Data preprocessing, augmentation strategies, model architectures, and hyperparameter choices
- **Agricultural Insights**: Area quantification for total agricultural land, tree canopy coverage, and sand deposition extent

### 4. Integrated Inference Pipeline

End-to-end system for processing new orthomosaics:
- Preprocessing module (tiling, normalization)
- Segmentation inference with mask generation
- Detection inference with bounding box output
- Post-processing (confidence thresholding, NMS, mask refinement)
- Output generation:
  - GeoTIFF segmentation masks with class labels
  - GeoJSON or Shapefile with detection bounding boxes
  - CSV summary table with area statistics per class

### 5. Technical Documentation
- **README**: Setup instructions, environment requirements, usage examples
- **Code Documentation**: Inline comments and docstrings
- **Requirements File**: All dependencies with version specifications
- **Deployment Guide**: Instructions for model deployment in production environments

## Quality Assurance

### Model Performance Targets

**Segmentation:**
- Mean IoU ≥ 0.75 across all classes
- Per-class IoU ≥ 0.70 for major classes (Agro Lands, Trees Canopy)
- Pixel accuracy ≥ 0.85

**Detection:**
- mAP@0.5 ≥ 0.70 overall
- Per-class AP ≥ 0.65 for all classes
- Inference time < 5 seconds per 1024x1024 tile on GPU

## Key Considerations

**Class Definition Ambiguities**: Establish clear criteria distinguishing between overlapping categories (e.g., scattered trees vs. tree canopy, fallow land vs. sand deposition).

**Class Imbalance**: Sand deposition may be rare compared to agricultural lands. Implement strategies such as focal loss, class weighting, or oversampling minority classes.

**Scale Variations**: Agricultural features vary dramatically in size (small farm plots vs. large forest canopies). Multi-scale architectures or pyramid approaches may be necessary.

**Geospatial Accuracy**: Maintain coordinate reference system throughout pipeline to enable accurate area calculations and geographic analysis.
