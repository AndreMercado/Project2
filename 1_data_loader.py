import os

DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28
FACE_WIDTH = 60
FACE_HEIGHT = 70

# Actual filenames in the data directories
_DIGIT_FILES = {
    'train':      ('trainingimages',      'traininglabels'),
    'validation': ('validationimages',    'validationlabels'),
    'test':       ('testimages',          'testlabels'),
}
_FACE_FILES = {
    'train':      ('facedatatrain',       'facedatatrainlabels'),
    'validation': ('facedatavalidation',  'facedatavalidationlabels'),
    'test':       ('facedatatest',        'facedatatestlabels'),
}


def load_images(filepath, width, height):
    """Parse a text file of concatenated images into a list of 2D char grids."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    num_images = len(lines) // height
    images = []
    for i in range(num_images):
        grid = []
        for r in range(height):
            line = lines[i * height + r].rstrip('\n').rstrip('\r')
            line = line.ljust(width)
            grid.append(line)
        images.append(grid)
    return images


def load_labels(filepath):
    """Parse a label file into a list of ints."""
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f if line.strip() != '']


def load_dataset(data_dir, dataset_type, split):
    """
    Load images and labels for a given dataset type and split.
    dataset_type: 'digits' or 'faces'
    split: 'train', 'validation', or 'test'
    Returns (images, labels)
    """
    if dataset_type == 'digits':
        width, height = DIGIT_WIDTH, DIGIT_HEIGHT
        img_file, lbl_file = _DIGIT_FILES[split]
    else:
        width, height = FACE_WIDTH, FACE_HEIGHT
        img_file, lbl_file = _FACE_FILES[split]

    img_path = os.path.join(data_dir, img_file)
    lbl_path = os.path.join(data_dir, lbl_file)

    images = load_images(img_path, width, height)
    labels = load_labels(lbl_path)

    assert len(images) == len(labels), (
        f"Mismatch: {len(images)} images vs {len(labels)} labels in {img_path}"
    )
    return images, labels
