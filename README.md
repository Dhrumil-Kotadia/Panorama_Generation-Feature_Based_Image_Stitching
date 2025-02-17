# Corner Detection and Image Stitching
Create a panoramic image from multiple images using Feature detection, Non Maximum Suppression, Feature Matching using feature descriptors and RANSAC.
<p align="center">
 <img src="stitched_images/Set2.png" width="700"/>
</p>

<p align="center">
 <img src="stitched_images/Set4.png" width="700"/>
</p>

<p align="center">
 <img src="stitched_images/Set3.png" width="700"/>
</p>


## Dependencies
Ensure you have the following dependencies installed:
```bash
pip install numpy opencv-python matplotlib scikit-image
```
## Pipeline
This project implements corner detection, feature matching, and homography estimation for image stitching using Python and OpenCV. The main steps include:
- **Corner Detection**: Harris Corner Detection and Adaptive Non-Maximal Suppression (ANMS)
<p float="center">
 <img src="outputs/0_Image1_corners.png" width="400"/>
 <img src="outputs/0_Image2_corners.png" width="400"/>
</p>
<p float="left">
 <img src="outputs/0_Image1_anms.png" width="400"/>
 <img src="outputs/0_Image2_anms.png" width="400"/>
</p>

- **Feature Description and Matching**: Generating descriptors and matching features using Euclidean distance
<p float="left">
 <img src="outputs/0_matching.png" width="400"/>
 <img src="outputs/1_matching.png" width="400"/>
</p>

- **Homography Estimation**: Using RANSAC to filter outliers and compute the transformation matrix
<p float="left">
 <img src="outputs/0_Ransac_Output.png" width="400"/>
 <img src="outputs/1_Ransac_Output.png" width="400"/>
</p>

- **Image Stitching**: Warping and blending images to create a panorama
<p align="center">
 <img src="outputs/mypano.png" width="700"/>
</p>

## Features
- Harris Corner Detection
- Adaptive Non-Maximal Suppression (ANMS)
- Feature Descriptor Extraction
- Feature Matching with Euclidean Distance
- Homography Estimation using Direct Linear Transform (DLT)
- RANSAC for Outlier Removal
- Image Warping and Blending

## Dependencies
Ensure you have the following dependencies installed:
```bash
pip install numpy opencv-python matplotlib scikit-image
```

## Usage
### Running the Code
Use the script to process images and perform image stitching:
```bash
python3 wrapper.py --Set 3
```

### Arguments
- `--Set`: The Set of Images to be used. Choose 1,2 or 3

## Author
Dhrumil Kotadia

## License
This project is licensed under the MIT License.


