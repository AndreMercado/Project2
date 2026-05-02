import os

# Step 1: Define the exact pixel dimensions of each image type.
# Digit images are 28 pixels wide and 28 pixels tall.
# Face images are 60 pixels wide and 70 pixels tall.
DIGIT_WIDTH  = 28
DIGIT_HEIGHT = 28
FACE_WIDTH   = 60
FACE_HEIGHT  = 70

# Step 2: Map each split name to the actual filenames on disk.
# Each dataset has three splits: train, validation, and test.
# Each split has two files: one for the images, one for the labels.
DIGIT_FILES = {
    'train':      ('trainingimages',     'traininglabels'),
    'validation': ('validationimages',   'validationlabels'),
    'test':       ('testimages',         'testlabels'),
}
FACE_FILES = {
    'train':      ('facedatatrain',      'facedatatrainlabels'),
    'validation': ('facedatavalidation', 'facedatavalidationlabels'),
    'test':       ('facedatatest',       'facedatatestlabels'),
}


def parse_single_image(all_lines, image_index, width, height):
    # Step 3: Extract one image from the file.
    # All images are stacked in one file, each taking exactly 'height' lines.
    # So image 0 starts at line 0, image 1 starts at line 'height', and so on.
    image_rows = []
    for row in range(height):

        # Step 4: Calculate which line in the file belongs to this row of this image.
        line_index = image_index * height + row
        line = all_lines[line_index]

        # Step 5: Clean up the line.
        # Remove newline characters left over from reading the file.
        line = line.rstrip('\n')
        line = line.rstrip('\r')

        # Step 6: Pad short lines with spaces so every row is exactly 'width' characters.
        line = line.ljust(width)

        image_rows.append(line)

    return image_rows


def load_images(filepath, width, height):
    # Step 7: Read the entire image file into memory as a list of lines.
    with open(filepath, 'r') as image_file:
        all_lines = image_file.readlines()

    # Step 8: Calculate how many images are in the file.
    # Since each image takes exactly 'height' lines, divide total lines by height.
    num_images = len(all_lines) // height
    images = []

    # Step 9: Parse each image one at a time and collect them into a list.
    for image_index in range(num_images):
        image = parse_single_image(all_lines, image_index, width, height)
        images.append(image)

    return images


def load_labels(filepath):
    # Step 10: Read the label file. Each line holds one integer label.
    # For digits: 0-9. For faces: 0 (not a face) or 1 (face).
    with open(filepath, 'r') as label_file:
        labels = []
        for line in label_file:
            stripped = line.strip()

            # Step 11: Skip blank lines that may appear at the end of the file.
            if stripped:
                labels.append(int(stripped))

    return labels


def load_dataset(data_dir, dataset_type, split):
    # Step 12: Choose the correct image dimensions and filenames
    # based on whether we are loading digits or faces.
    is_digits = dataset_type == 'digits'

    if is_digits:
        width, height = DIGIT_WIDTH, DIGIT_HEIGHT
        image_filename, label_filename = DIGIT_FILES[split]
    else:
        width, height = FACE_WIDTH, FACE_HEIGHT
        image_filename, label_filename = FACE_FILES[split]

    # Step 13: Build the full file paths and load images and labels.
    images = load_images(os.path.join(data_dir, image_filename), width, height)
    labels = load_labels(os.path.join(data_dir, label_filename))

    # Step 14: Return both lists. Image at index i matches label at index i.
    return images, labels
