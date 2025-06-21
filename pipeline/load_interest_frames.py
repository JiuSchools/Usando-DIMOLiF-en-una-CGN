# Encuentra los frames de interes IF's
import pandas as pd
from collections import defaultdict

def load_if_data_from_csv(csv_path):
    """
    Carga los Interest Frames desde el CSV y los agrupa por video y por frame.

    Args:
        csv_path (str): ruta al archivo CSV

    Returns:
        dict: diccionario {video_name: {frame: [(x, y), ...]}}
    """
    df = pd.read_csv(csv_path)

    # Agrupar por video y luego por frame
    video_dict = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        video = row['video']
        frame = int(row['frame'])
        x = int(row['x'])
        y = int(row['y'])
        video_dict[video][frame].append((x, y))

    return video_dict
