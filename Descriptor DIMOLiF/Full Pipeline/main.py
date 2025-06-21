from extract_save_histograms import extract_and_save_histograms
from extract_save_histograms_skip import extract_and_save_histograms_skip_existing

extract_and_save_histograms_skip_existing(
    csv_path="/content/drive/MyDrive/Validation_STIP_csv/interest_frames_ALL.csv",
    video_dir="/content/drive/MyDrive/aaa/validacion/carpeta3",
    output_dir="/content/drive/MyDrive/Sets_Numpy/Validation_Numpy/",
    T=8,
    threshold=3,
    num_bins=8,
    bandwidth=0.2,
    kde_threshold=0.05
)