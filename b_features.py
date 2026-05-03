GRID_ROWS: int = 7
GRID_COLS: int = 7


def pixel_is_filled(character: str) -> int:
    # Step 19: Check if a character represents a drawn pixel.
    # A space means the pixel is empty. Anything else ('+', '#', etc.) means filled.
    return int(character != ' ')


def get_pixel(image: list[str], row: int, column: int) -> str:
    # Step 25: Safely read one character from the image grid.
    # Some lines may be shorter than expected, so return a space if out of bounds.
    column_is_in_bounds: bool = column < len(image[row])
    if column_is_in_bounds:
        return image[row][column]
    return ' '


def compute_pixel_features(image: list[str]) -> list[int]:
    # Step 18: Convert the entire image into a flat list of 0s and 1s.
    # Each pixel becomes one feature: 1 if filled, 0 if empty.
    # A 28x28 digit image produces 784 features.
    # A 60x70 face image produces 4200 features.
    features: list[int] = []
    for row in image:
        for character in row:
            features.append(pixel_is_filled(character))
    return features


def count_filled_in_block(image: list[str], row_start: int, row_end: int, col_start: int, col_end: int) -> tuple[int, int]:
    # Step 24: Count how many pixels are filled inside one rectangular block.
    # Also count the total pixels in the block so we can compute a fraction later.
    filled_count: int = 0
    total_count:  int = 0
    for row_index in range(row_start, row_end):
        for col_index in range(col_start, col_end):
            character: str = get_pixel(image, row_index, col_index)
            filled_count += pixel_is_filled(character)
            total_count  += 1
    return filled_count, total_count


def compute_grid_features(image: list[str]) -> list[float]:
    # Step 20: Divide the image into a 7x7 grid of rectangular blocks.
    # Each block produces one feature: the fraction of filled pixels inside it.
    # This gives 49 features that capture the rough shape and layout of the image.
    height: int           = len(image)
    width:  int           = len(image[0]) if image else 0
    features: list[float] = []

    for block_row in range(GRID_ROWS):
        # Step 21: Calculate the pixel row range for this block row.
        row_start: int = (block_row * height) // GRID_ROWS
        row_end:   int = ((block_row + 1) * height) // GRID_ROWS

        for block_col in range(GRID_COLS):
            # Step 22: Calculate the pixel column range for this block column.
            col_start: int = (block_col * width) // GRID_COLS
            col_end:   int = ((block_col + 1) * width) // GRID_COLS

            # Step 23: Count filled pixels in this block.
            filled_count, total_count = count_filled_in_block(
                image, row_start, row_end, col_start, col_end
            )

            # Step 26: Compute the fill fraction. Guard against empty blocks.
            block_is_empty: bool = total_count == 0
            if block_is_empty:
                fill_fraction: float = 0.0
            else:
                fill_fraction = filled_count / total_count

            features.append(fill_fraction)

    return features


def extract_features(image: list[str]) -> list[float]:
    # Step 17: Combine both feature types into one single vector.
    # Pixel features capture exact detail. Grid features capture overall structure.
    # Together they give the classifier more information to work with.
    return compute_pixel_features(image) + compute_grid_features(image)
