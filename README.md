# Football Player Prediction

This project uses a YOLO model to detect and track football players in video footage. The project includes scripts for extracting frames from a video, training a YOLO model, and testing the model on new video footage.

Example uses 'portugal_vs_spain.mp4' and using  https://github.com/HumanSignal/labelImg, the images were labeled specifically for YOLO model.

You can see the result in `cr7_tracked.mp4`

# How to use
execute `extract_frames.py`, then execute `model_test.py`. There will be a result file generated `cr7_tracked.mp4`. Run the file to see marked player.

## Project Structure

### `extract_frames.py`

This scripts extracts every 10th frame from the video file. The files are then used for labeling. 
The frames can be found in 'frames' Folder.

yolo_files contains labeled images with weights used for training and validation.

### `split_dataset.py`

This script splits the dataset into training, validation, and testing sets. It does so by:

- Creating folders: It creates separate directories for train, val, and test for both images and their corresponding labels.
- Listing images: It gathers all .jpg files from your images folder.
- Shuffling and splitting: It randomly shuffles the image filenames and then splits them into 70% training, 20% validation, and 10% testing.
- Moving files: It moves each image (and its corresponding label file) into the appropriate subfolder.

### `yolo_model_training.py`

This script is used for training the yolo model

### `model_test.py`

This script is used to create a new .mp4 file with marked player.
