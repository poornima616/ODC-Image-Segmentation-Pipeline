# ODC-Image-Segmentation-Pipeline

The project follows a structured, end-to-end workflow designed to create a high-quality, annotated Indian road dataset suitable for computer vision research and model development. The pipeline consists of the following stages:

1. Image Collection

Street-level images are collected from Mapillary, covering a wide variety of Indian road environments including highways, urban streets, rural roads, and mixed-traffic scenarios. The collection ensures diversity in geography, lighting, weather, and scene complexity.

2. Data Cleaning and Filtering

Raw images undergo quality filtering to remove irrelevant, blurred, or unusable frames. Perceptual hashing techniques (aHash, pHash, dHash) are applied to detect and eliminate duplicate or near-duplicate images. This maintains dataset consistency and avoids redundancy.

3. Segmentation and Annotation Using SAM-ViT-H

Cleaned images are processed with the Segment Anything Model (SAM-ViT-H).
This step includes:

Automatic mask generation for road elements such as vehicles, lanes, pedestrians, and road boundaries

Extraction of contours and polygon coordinates using OpenCV

Filtering masks based on area thresholds to retain only meaningful segments

Organizing segmentation output into structured annotation formats

The combination of SAM automation and manual validation ensures accurate and reliable annotations.

4. Metadata Structuring and Export

For each image, segmentation regions and metadata (file name, size, labels, polygon coordinates, GPS information when available) are compiled. The data is exported in both:

VGG VIA JSON format

Excel spreadsheets

This formatting ensures compatibility with common training pipelines, annotation tools, and visualization frameworks.

5. Dataset Validation

A manual review phase ensures the correctness of segmentation masks, removal of erroneous outputs, and verification of class labels. This step maintains annotation quality and adherence to dataset standards.

6. Final Dataset Preparation

After validation, the dataset becomes a clean, diverse, and well-annotated resource capturing real-world Indian road conditions. The resulting workflow is scalable and repeatable, allowing further expansion across regions and use cases.
