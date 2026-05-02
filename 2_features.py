"""
Two feature representations for image classification.

Feature 1 - Raw binary pixels:
  One binary value per pixel: 1 if the character is not a space, 0 otherwise.
  For a 28x28 digit image this produces a 784-dimensional vector.

Feature 2 - Grid block counts:
  Divide the image into a GRID_R x GRID_C grid of rectangular blocks.
  Each block contributes one value: the fraction of filled pixels in that block.
  This produces a GRID_R*GRID_C-dimensional vector (floats in [0,1]).
"""

GRID_R = 7
GRID_C = 7


def _is_filled(ch):
    return 0 if ch == ' ' else 1


def pixel_features(image):
    """Return flat list of binary pixel values."""
    features = []
    for row in image:
        for ch in row:
            features.append(_is_filled(ch))
    return features


def grid_features(image):
    """Return fractional fill per grid block."""
    height = len(image)
    width = len(image[0]) if image else 0
    features = []
    for br in range(GRID_R):
        r_start = (br * height) // GRID_R
        r_end = ((br + 1) * height) // GRID_R
        for bc in range(GRID_C):
            c_start = (bc * width) // GRID_C
            c_end = ((bc + 1) * width) // GRID_C
            total = 0
            filled = 0
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    ch = image[r][c] if c < len(image[r]) else ' '
                    total += 1
                    filled += _is_filled(ch)
            features.append(filled / total if total > 0 else 0.0)
    return features


def extract_features(image):
    """Concatenate pixel features and grid features into one vector."""
    return pixel_features(image) + grid_features(image)
