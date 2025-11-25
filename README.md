# GalaxyAI - Automated Galaxy Morphology Classification System

[cite_start]This repository contains an end-to-end Machine Learning pipeline for classifying galaxy images into morphological categories, with cloud deployment, monitoring, and retraining capabilities. [cite: 995, 997, 998]

## 1. Project Overview

The GalaxAI project is an MLOps demonstration in the field of astrophysics, focused on automated image classification.

| Field            | Description                                                                                                                                                                                              |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Project Name** | [cite_start]Galaxy AI - Automated Galaxy Morphology Classification System [cite: 997]                                                                                                                    |
| **Objective**    | [cite_start]Demonstrate an end-to-end Machine Learning pipeline for classifying galaxy images into morphological categories, with cloud deployment, monitoring, and retraining capabilities. [cite: 998] |
| **Domain**       | [cite_start]Astronomy / Astrophysics [cite: 999]                                                                                                                                                         |
| **Data Type**    | [cite_start]RGB Galaxy Images (Non-tabular) [cite: 1000]                                                                                                                                                 |
| **Model Type**   | [cite_start]Multi-class Classification (10 classes) [cite: 1001]                                                                                                                                         |

---

## 2. Problem Statement

Galaxy morphology classification is fundamental to understanding galaxy formation and evolution. Traditionally, this is a labor-intensive task performed by human volunteers (e.g., via Galaxy Zoo). [cite_start]This project automates galaxy classification using deep learning, enabling rapid, consistent, and scalable morphological analysis through a complete MLOps pipeline. [cite: 1003, 1004]

### Classification Categories (10 Classes)

The system classifies galaxies into one of 10 categories, each represented by a class ID:

| Class | Label                     | Description                                                         |
| :---: | :------------------------ | :------------------------------------------------------------------ |
| **0** | **Disturbed**             | [cite_start]Galaxies showing gravitational disturbance [cite: 1006] |
| **1** | **Merging**               | [cite_start]Two or more galaxies merging [cite: 1006]               |
| **2** | **Round Smooth**          | [cite_start]Elliptical with round, smooth appearance [cite: 1006]   |
| **3** | **In-between Smooth**     | [cite_start]Intermediate roundness elliptical [cite: 1006]          |
| **4** | **Cigar-Shaped**          | [cite_start]Elongated elliptical galaxies [cite: 1006]              |
| **5** | **Barred Spiral**         | [cite_start]Spiral with central bar structure [cite: 1006]          |
| **6** | **Unbarred Tight Spiral** | [cite_start]Tightly wound spiral arms, no bar [cite: 1006]          |
| **7** | **Unbarred Loose Spiral** | [cite_start]Loosely wound spiral arms, no bar [cite: 1006]          |
| **8** | **Edge-on No Bulge**      | [cite_start]Disk galaxy edge-on, no bulge [cite: 1006]              |
| **9** | **Edge-on With Bulge**    | [cite_start]Disk galaxy edge-on, visible bulge [cite: 1006]         |

---

## 3. Data Sources

### Primary Dataset: Galaxy10 DECaLS

| Specification          | Value                                                               |
| :--------------------- | :------------------------------------------------------------------ |
| [cite_start]**Source** | astroNN Galaxy10 DECaLS Dataset [cite: 1009]                        |
| [cite_start]**URL**    | https://astronn.readthedocs.io/en/latest/galaxy10.html [cite: 1010] |
| **Image Size**         | [cite_start]256×256 pixels (RGB) [cite: 1012]                       |
| **Format**             | [cite_start]HDF5 file [cite: 1013]                                  |
| **Total Images**       | [cite_start]17,736 [cite: 1014]                                     |
| **Classes**            | [cite_start]10 morphological categories [cite: 1015]                |

### Data Access Code

[cite_start]The dataset can be loaded directly using the `astroNN` library in Python: [cite: 1034]

````python
from astroNN.datasets import galaxy10
images, labels = galaxy10.load_data()
# images.shape = (17736, 256, 256, 3)
# labels.shape = (17736,)
[cite_start]``` [cite: 1036, 1037, 1038]
````
