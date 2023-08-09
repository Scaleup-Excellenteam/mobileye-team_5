import json
from typing import Dict, Any, List, Union

import pandas as pd
from PIL import Image
from shapely.geometry import Polygon
from pandas import DataFrame
from pathlib import Path
from typing import List


SEQ_IMAG: str = 'seq_imag'  # Serial number of the image
NAME: str = 'name'
IMAG_PATH: str = 'imag_path'
GTIM_PATH: str = 'gtim_path'
JSON_PATH: str = 'json_path'
X: str = 'x'
Y: str = 'y'
COLOR: str = 'color'
# The label we wanna look for in the polygons json file
TFL_LABEL = ['traffic light']
POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
# # Data CSV columns:
CSV_INPUT: List[str] = [SEQ_IMAG, NAME, IMAG_PATH, JSON_PATH, GTIM_PATH]
CSV_OUTPUT: List[str] = [SEQ_IMAG, NAME, IMAG_PATH, JSON_PATH, GTIM_PATH, X, Y, COLOR]

TRAIN_TEST_VAL = 'train_test_val'
TRAIN = 'train'
TEST = 'test'
VALIDATION = 'validation'

BASE_SNC_DIR = Path.cwd()
DATA_DIR: Path = (BASE_SNC_DIR / 'data')
FULL_IMAGES_DIR: Path = 'fullImages'  # Where we write the full images
CROP_DIR: Path = DATA_DIR / 'crops'
PART_IMAGE_SET: Path = DATA_DIR / 'images_set'
IMAGES_1: Path = PART_IMAGE_SET / 'Image_1'

# # Crop size:
DEFAULT_CROPS_W: int = 32
DEFAULT_CROPS_H: int = 96

SEQ: str = 'seq'  # The image seq number -> for tracing back the original image
IS_TRUE: str = 'is_true'  # Is it a traffic light or not.
IS_IGNORE: str = 'is_ignore'
# investigate the reason after
CROP_PATH: str = 'path'
X0: str = 'x0'  # The bigger x value (the right corner)
X1: str = 'x1'  # The smaller x value (the left corner)
Y0: str = 'y0'  # The smaller y value (the lower corner)
Y1: str = 'y1'  # The bigger y value (the higher corner)
COL: str = 'col'

RELEVANT_IMAGE_PATH: str = 'path'
ZOOM: str = 'zoom'  # If you zoomed in the picture, then by how much? (0.5. 0.25 etc.).
PATH: str = 'path'

# # CNN input CSV columns:
CROP_RESULT: List[str] = [SEQ, IS_TRUE, IS_IGNORE, CROP_PATH, X0, X1, Y0, Y1, COL]
ATTENTION_RESULT: List[str] = [RELEVANT_IMAGE_PATH, X, Y, ZOOM, COL]

# # Files path
BASE_SNC_DIR: Path = Path.cwd().parent
ATTENTION_PATH: Path = DATA_DIR / 'attention_results'

ATTENTION_CSV_NAME: str = 'attention_results.csv'
CROP_CSV_NAME: str = 'crop_results.csv'

MODELS_DIR: Path = DATA_DIR / 'models'  # Where we explicitly copy/save good checkpoints for "release"
LOGS_DIR: Path = MODELS_DIR / 'logs'  # Each model will have a folder. TB will show all models


# # File names (directories to be appended automatically)
TFLS_CSV: str = 'tfls.csv'
CSV_OUTPUT_NAME: str = 'results.csv'

def make_crop(x, y, color, zoom, *args, **kwargs):
    """
    The function that creates the crops from the image.
    Your return values from here should be the coordinates of the crops in this format (x0, x1, y0, y1, crop content):
    'x0'  The bigger x value (the right corner)
    'x1'  The smaller x value (the left corner)
    'y0'  The smaller y value (the lower corner)
    'y1'  The bigger y value (the higher corner)
    """
    # Define the size of the crop region around the TFL
    crop_width = DEFAULT_CROPS_W
    crop_height = DEFAULT_CROPS_H
    y0_offset = 0
    y1_offset = 0
    # Adjust y0 offset and y1 offset based on color
    if color == 'r':
        y0_offset = 1 / 3
        y1_offset = 2 / 3
    elif color == 'g':
        y0_offset = 2 / 3
        y1_offset = 1 / 3

    # Calculate the default cropping region around the TFL
    x0 = int(x - crop_width // 2)
    x1 = int(x + crop_width // 2)
    y0 = int(y - (crop_height * y0_offset))
    y1 = int(y + (crop_height * y1_offset))

    return x0, x1, y0, y1, 'crop_data'


def check_crop(image_json_path, x0, x1, y0, y1):
    image_json_path = PART_IMAGE_SET / IMAGES_1 / image_json_path

    image_json = json.load(Path(image_json_path).open())
    traffic_light_polygons: List[POLYGON_OBJECT] = [image_object for image_object in image_json['objects']
                                                    if image_object['label'] in TFL_LABEL]
    is_true, ignore = False, False
    cropped_polygon = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    count_traffic_lights = 0
    for tl_polygon in traffic_light_polygons:
        if cropped_polygon.contains(Polygon(tl_polygon['polygon'])):
            count_traffic_lights += 1
            if count_traffic_lights >= 2:
                ignore = True
                break
            is_true = True

    return is_true, ignore


def save_for_part_2(crops_df: DataFrame):
    """
    *** No need to touch this. ***
    Saves the result DataFrame containing the crops data in the relevant folder under the relevant name for part 2.
    """
    if not ATTENTION_PATH.exists():
        ATTENTION_PATH.mkdir()
    crops_sorted: DataFrame = crops_df
    crops_sorted.to_csv(ATTENTION_PATH / CROP_CSV_NAME, index=False)


def create_crops(df: DataFrame, IGNOR=None) -> DataFrame:
    # Your goal in this part is to take the coordinates you have in the df, run on it, create crops from them, save them
    # in the 'data' folder, then check if crop you have found is correct (meaning the TFL is fully contained in the
    # crop) by comparing it to the ground truth and in the end right all the result data you have in the following
    # DataFrame (for doc about each field and its input, look at 'CROP_RESULT')
    #
    # *** IMPORTANT ***
    # All crops should be the same size or smaller!!!
    # Run this from your 'code' folder so that it will be in the right relative folder from your data folder.

    # creates a folder for you to save the crops in, recommended not must
    if not CROP_DIR.exists():
        CROP_DIR.mkdir()

    # For documentation about each key end what it means, click on 'CROP_RESULT' and see for each value what it means.
    # You wanna stick with this DataFrame structure because its output is the same as the input for the next stages.
    result_df = pd.DataFrame(columns=CROP_RESULT)

    # A dict containing the row you want to insert into the result DataFrame.
    result_template: Dict[Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', JSON_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                  COL: ''}
    for index, row in df.iterrows():
        # Save sequence TFL in each image
        result_template[SEQ] = row[SEQ]
        # Save color TFL in each image
        result_template[COL] = row[COL]

        # Extract image_path
        image_path = row[CROP_PATH]

        # Extrac corp rect from
        x0, x1, y0, y1, crop = make_crop(row[X], row[Y], row[COL], row[ZOOM])
        result_template[X0], result_template[X1], result_template[Y0], result_template[Y1] = x0, x1, y0, y1

        # crop.save(CROP_DIR / crop_path)

        # Save json path (Anton need to pass from part 1)
        image_json_path = image_path.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        result_template[JSON_PATH] = image_json_path

        # Check crop rectangle if it TFL or not, ignore if it parts of TFL, double TFL,
        result_template[IS_TRUE], result_template[IS_IGNORE] = check_crop(image_json_path, x0, x1, y0, y1)

        # Create unique path for crop TFL
        if result_template[IS_IGNORE]:
            tag = 'i'
        else:
            tag = 'T' if result_template[IS_TRUE] else 'F'

        crop_path = f'/data/crops/{image_path[:-4]}_{row[COL]}{tag}_{index}'

        # Save unique path
        result_template[CROP_PATH] = crop_path

        # Create a DataFrame with the current result_template data
        result_row_df = pd.DataFrame(result_template, index=[index])

        # Concatenate the current row DataFrame with the existing result DataFrame
        result_df = pd.concat([result_df, result_row_df], ignore_index=True)
        if result_template[IS_TRUE] or not result_template[IS_IGNORE]:
            # Extract image_path and open the image
            image_path = row[CROP_PATH]
            image = Image.open(PART_IMAGE_SET / IMAGES_1 / image_path)
            # Crop the image using the coordinates
            cropped_image = image.crop((x0, y0, x1, y1))
            # Save cropped image
            full_path = CROP_DIR / f'{image_path[:-4]}_{row[COL]}{tag}_{index}.png'
            print(f"Saving to: {full_path}")
            cropped_image.save(full_path)

    # A Short function to help you save the whole thing - your welcome ;)
    save_for_part_2(result_df)
    return result_df


def create_all_crops():
    df = pd.read_csv(BASE_SNC_DIR/ATTENTION_PATH/ATTENTION_CSV_NAME)
    crop_df = create_crops(df)


