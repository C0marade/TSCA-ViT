# Data Preparing

1. In this project, we utilize a dataset containing .png images and their corresponding segmentation labels. The dataset is organized into separate folders for images and labels, with file names specified in a .txt file for each split (e.g., train.txt, test.txt). 
Below is a detailed explanation of the dataset structure and the preprocessing steps applied:

Images: Stored in the images/ directory, representing the raw input data in grayscale format (.png files).
Labels: Stored in the labels/ directory, representing the segmentation masks (.png files).
Text files (train.txt, test.txt) containing the base names of the files to be used in each split.

2. Preprocessing Steps
Data Loading:

Each image is loaded from the images/ folder and converted to a grayscale numpy array using PIL.Image.open.
Corresponding segmentation labels are loaded from the labels/ folder as numpy arrays.
Image Augmentation:

For training data, augmentation techniques are applied to increase the diversity of the dataset and prevent overfitting. These include:
Flipping: Vertical and horizontal flips are applied with a probability of 50%.
Rotation: Random rotations between -40° and 40° are performed.
Scaling: Images are randomly scaled with factors between 0.5 and 2.
Gaussian Noise: Noise is added to simulate realistic imperfections.
Gaussian Blur: Blurring is applied to smooth the images.
Contrast Adjustment: Random contrast adjustments enhance the variety of visual characteristics.
Affine Transformations: Include shear, translation, and piecewise affine distortions.
Mask Preprocessing:

Segmentation masks are converted into one-hot encoded representations for multi-class segmentation. This ensures compatibility with the model's expected input format.
Resizing:

Both images and labels are resized to the target resolution (img_size) using bilinear interpolation for images and nearest-neighbor interpolation for labels. This standardizes input dimensions for model training.
Tensor Conversion:

Images are converted into PyTorch tensors and normalized to float32. Labels are converted into tensors with a long data type, which is required for segmentation tasks.
Random Transformations:

Custom random rotation and flipping operations are implemented to augment the dataset further:
Random rotation by 90°, 180°, or 270°.
Random horizontal or vertical flipping.
3. The directory structure of the whole project is as follows:

```bash
.
data_dir/
    images/
        0001.png
        0002.png
    labels/
        0001.png
        0002.png
    train.txt
    test.txt


