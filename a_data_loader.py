import os

# Digit images are 28 pixels wide and 28 pixels tall.
# Face images are 60 pixels wide and 70 pixels tall.
DIGIT_WIDTH:  int = 28
DIGIT_HEIGHT: int = 28
FACE_WIDTH:   int = 60
FACE_HEIGHT:  int = 70

# Each dataset has three splits: train, validation, and test.
# Each split has two files: one for the images, one for the labels.
DIGIT_FILES: dict[str, tuple[str, str]] = {
    'train':      ('trainingimages',     'traininglabels'),
    'validation': ('validationimages',   'validationlabels'),
    'test':       ('testimages',         'testlabels'),
}
FACE_FILES: dict[str, tuple[str, str]] = {
    'train':      ('facedatatrain',      'facedatatrainlabels'),
    'validation': ('facedatavalidation', 'facedatavalidationlabels'),
    'test':       ('facedatatest',       'facedatatestlabels'),
}


def parse_single_image(all_lines: list[str], image_index: int, width: int, height: int) -> list[str]:
    # Step 10: Extract one image from the file.
    # All images are stacked in one file, each taking exactly 'height' lines.
    # So image 0 starts at line 0, image 1 starts at line 'height', and so on.
    image_rows: list[str] = []
    for row in range(height):

        # Step 11: Calculate which line in the file belongs to this row of this image.
        line_index: int = image_index * height + row
        line: str       = all_lines[line_index]

        # Step 12: Clean up the line.
        # Remove newline characters left over from reading the file.
        line = line.rstrip('\n')
        line = line.rstrip('\r')

        # Step 13: Pad short lines with spaces so every row is exactly 'width' characters.
        line = line.ljust(width)

        image_rows.append(line)

    return image_rows


def load_images(filepath: str, width: int, height: int) -> list[list[str]]:
    # Step 7: Read the entire image file into memory as a list of lines.
    with open(filepath, 'r') as image_file:
        all_lines: list[str] = image_file.readlines()

    # Step 8: Calculate how many images are in the file.
    # Since each image takes exactly 'height' lines, divide total lines by height.
    num_images: int         = len(all_lines) // height
    images: list[list[str]] = []

    # Step 9: Parse each image one at a time and collect them into a list.
    for image_index in range(num_images):
        image: list[str] = parse_single_image(all_lines, image_index, width, height)
        images.append(image)

    return images


def load_labels(filepath: str) -> list[int]:
    # Step 14: Read the label file. Each line holds one integer label.
    # For digits: 0-9. For faces: 0 (not a face) or 1 (face).
    with open(filepath, 'r') as label_file:
        labels: list[int] = []
        for line in label_file:
            stripped: str = line.strip()

            # Step 15: Skip blank lines that may appear at the end of the file.
            line_has_content: bool = bool(stripped)
            if line_has_content:
                labels.append(int(stripped))

    return labels


def load_dataset(data_dir: str, dataset_type: str, split: str) -> tuple[list[list[str]], list[int]]:
    # Step 5: Choose the correct image dimensions and filenames
    # based on whether we are loading digits or faces.
    is_digits: bool = dataset_type == 'digits'

    if is_digits:
        width: int          = DIGIT_WIDTH
        height: int         = DIGIT_HEIGHT
        image_filename: str = DIGIT_FILES[split][0]
        label_filename: str = DIGIT_FILES[split][1]
    else:
        width  = FACE_WIDTH
        height = FACE_HEIGHT
        image_filename = FACE_FILES[split][0]
        label_filename = FACE_FILES[split][1]

    # Step 6: Build the full file paths and load images and labels.
    images: list[list[str]] = load_images(os.path.join(data_dir, image_filename), width, height)
    labels: list[int]       = load_labels(os.path.join(data_dir, label_filename))

    # Step 16: Return both lists. Image at index i matches label at index i.
    return images, labels
