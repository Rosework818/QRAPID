# QRAPID

This code is used for the color recognition application developed for "Smart DNA Hydrogel-Responsive Encode Enables Portable Nucleic Acid Detection”

Below is a quick guide to using the UI of the program random_forest.py, organized by function and explained step-by-step in a concise manner.

## Architecture:<br>
<div align=center><img src="" width="740" height="437"/></div><br>

## Overall Interface
Window Title: Pathogen Detection Color QR Code Analysis Software.

Left Side: Control Panel (buttons, progress bar, system status).

Right Side: Display Area (Notebook with three tabs: Image Display, Independent Analysis Results, Performance Analysis).

Supported Image Formats: .jpg, .jpeg, .png, .bmp, .tiff.

## Common Buttons and Workflow
Select Training Folder and Train Random Forest Model

Purpose: Automatically divide the training/validation from a folder containing training images and train a random forest.

Requirements: Training filenames or paths must include "yellow" (positive samples) or "blue" (negative samples). split_train_val will determine labels based on the presence of these strings.

Results: Saves random_forest_model.pkl, confusion_matrix.png, roc_curve.png, mean_bgr_hsv_lab.csv, fpr_tpr.csv, etc. Training information is printed in the console and displayed via a pop-up.

## Select Independent Analysis Image
Purpose: Choose a full image containing a QR code (the program will perform QR code localization and region extraction).

Result: The filename of the loaded image will be displayed in the status area at the bottom left.

## Output Independent Analysis Image and Table Results
Purpose: Run detection and region analysis on the loaded QR code image.

Results: The analysis results are displayed in the "Image Display" tab under "Independent Analysis Result Image/Enhanced Image". Each region's results are written into the table in the "Independent Analysis Results" tab, and text results are shown in the "Performance Analysis" area.

Select Single Circle Image / Output Single Circle Image and Table Results

Purpose: Analysis process for a single circular reaction zone (not a full image with a QR code). After selecting the image, click "Output" to display the processed image and prediction (Yellow/Blue) with probability.

## Save Independent Analysis Results
Purpose: Save the current analysis results as a CSV file (select the file location).

## Select Folder for System Validation

Purpose: Writes image paths from the folder into a "Independent Analysis Validation.csv" file in that folder (CSV header includes condition, image path, 5 true label columns, and 5 predict columns).

Note: You need to manually fill in the true labels for each image in the CSV (Control/H1N1/Rhinovirus/Flu B/RSV columns use 0/1).

## System Validation 
Purpose: Select a top-level folder containing validation images (folder names need to be parsable as temperature_distance_angle_device, e.g., "6500_30_0_phone1"). comprehensive_validation will:

Read images from the folder, detect and analyze each image's five regions.

Write results back into the "Independent Analysis Validation.csv" (filling the predict columns).

Calculate TP/FP/FN/TN, success rate, Precision, Recall, F1, and display results in the "Performance Analysis" tab with a pop-up notification of the processed quantity.

Note: If the folder name cannot be split into four segments (temperature_distance_angle_device), an error will occur and the program will exit.

## Batch Process Independent Analysis Images and Save Results

Purpose: Select a folder containing images. The program will perform independent analysis on all images in the directory and save the results as a CSV file ("Independent Analysis Results.csv").

## Batch Process Single Circle Images and Save Results

Purpose: Run single circle prediction on each circular image in the directory and save the results as a CSV file ("Single Circle Image Independent Analysis Results.csv").

## Result Display and Interpretation

Independent Analysis Results Table Columns: Region name, result, positive probability (yellow), negative probability (blue), decision logic.

Color Indicators (UI):

Positive (Yellow): Yellow background (#fff200)

Negative (Blue): Blue background (#0015ff)

Invalid (Invalid/Control Failure): Red background

Performance Analysis Page: Displays textual performance/log information (training/validation/single sample analysis output, etc.).

File Output Location: Training/validation-related files are saved in the current working directory or a user-selected directory (e.g., model file random_forest_model.pkl, confusion_matrix.png, roc_curve.png, various CSV files, etc.).
File/CVS Points

image_path_to_csv writes "Independent Analysis Validation.csv" in the selected folder, initially only writing condition and image path. True labels need to be manually or externally filled (Control/H1N1/Rhinovirus/Flu B/RSV columns use 0/1).

comprehensive_validation reads the CSV and matches using the image path column to write predict values back into the corresponding "{Region} predict" columns.

If using confusion matrix visualization (commented code), the confusion matrix image will be saved to the image path (the commented part no longer displays automatically).

## Common Issues and Suggestions

Training data must include "yellow" or "blue" identifiers, and there must be enough samples; otherwise, training or metric calculation will fail.

For system validation, the image path in the CSV must match the validation_results exactly (file name or relative path); otherwise, it will be skipped with a warning.

If the GUI is unresponsive or the progress bar does not update, ensure not to close the program during batch operations; the progress bar will attempt to call update_idletasks.

For debugging, observe console output (it will print detailed error/warning information).
