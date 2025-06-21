#Este codigo es un principal opcional, en caso de que la ejecución del codigo main.py se hallá detenido por
# cualquier situación este codigo permite la ejecución del programa sin tener que procesar videos ya 
# procesados.

import pandas as pd
from pathlib import Path
import os
from load_video import load_video_to_array
from STIP_computation import compute_derivatives, compute_harris_response, compute_structure_tensor, detect_interest_points
import csv

output_dir = "/content/drive/MyDrive/Sets/Train_STIP_P2/"
file_path = "/content/drive/MyDrive/Sets/Train_STIP_P2/interest_frames_all.csv"
VIDEOS = "/content/drive/MyDrive/Sets/Train_STIP_P2/"
df = pd.read_csv(file_path)
videos_processed = df['video'].unique().tolist()



videos_paths = list(Path(VIDEOS).rglob("*.mp4"))

for video in videos_paths:
  if os.path.basename(video) not in videos_processed:
      # Cargar video completo en escala [0, 1]
      print(f"Procesando video: {video}\n")
      try:
          video_array = load_video_to_array(video)
          if video_array is None:
              print("ERROR al cargar el video")
      except Exception as e:
          print(f"Error: {e}")

      T_total = video_array.shape[2]
      print(f"Frames cargados: {T_total}")

      # Parámetros
      sigma_l = 2
      tau_l = 2
      sigma_i = sigma_l * 2
      tau_i = tau_l * 2
      k = 0.004
      threshold = 0.01
      block_size = 20  # tamaño de bloque para ahorrar RAM

      puntos_detectados = []

      for t0 in range(0, T_total - block_size, block_size - 2):  # solapamiento para bordes
          t1 = min(t0 + block_size, T_total)
          video_block = video_array[:, :, t0:t1]

          Lx, Ly, Lt = compute_derivatives(video_block, sigma_l, tau_l)
          tensor = compute_structure_tensor(Lx, Ly, Lt, sigma_i, tau_i)
          H = compute_harris_response(tensor, k)
          puntos = detect_interest_points(H, threshold)

          for x, y, t_rel in puntos:
              t_abs = t0 + t_rel
              if 1 <= t_abs < T_total - 1:
                  puntos_detectados.append((x, y, t_abs))

        # Agrupar por frame y guardar
      from collections import defaultdict
      puntos_por_frame = defaultdict(list)
      for x, y, t in puntos_detectados:
          puntos_por_frame[t].append((x, y))

      # Guardar datos en un csv
      csv_output = os.path.join(output_dir, "interest_frames_all.csv")
      video_name = os.path.basename(video)

      # Append rows to CSV (create header if file does not exist)
      write_header = not os.path.exists(csv_output)

      with open(csv_output, "a", newline='') as csvfile:
          writer = csv.writer(csvfile)
          if write_header:
              writer.writerow(["video", "frame", "num_points_in_frame", "x", "y"])

          for t in sorted(puntos_por_frame.keys()):
              puntos = puntos_por_frame[t]
              for x, y in puntos:
                  writer.writerow([video_name, t, len(puntos), x, y])


      print(f"Se detectaron {len(puntos_detectados)} puntos en {len(puntos_por_frame)} frames.")
      """
      for t in sorted(puntos_por_frame.keys()):
           print(f"Frame {t}: {len(puntos_por_frame[t])} puntos")
       """
#    end = time.time()
#    print(f"Tiempo total: {end - begin:.2f} segundos")