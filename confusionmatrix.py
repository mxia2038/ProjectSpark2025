import numpy as np
import pandas as pd
from pretty_confusion_matrix import pp_matrix
import scienceplots  # 导入 Science Plots 库
import matplotlib.pyplot as plt


array = np.array([[3,  1,  0,  0,  0,  0],
                  [0,  4,  3,  0,  0,  0],
                  [0,  2,  5,  0,  0,  0],
                  [0,  0,  1,  3,  2,  0],
                  [0,  0,  0,  0,  5,  0],
                  [0,  0,  0,  0,  0,  1]])

# get pandas dataframe
df_cm = pd.DataFrame(array, index=range(0, 6), columns=range(0, 6))
# colormap: see this and choose your more dear
cmap = 'YlGn'

# Create figure first
fig, ax1 = plt.subplots(figsize=(12, 12), dpi=300)  # Higher DPI for print quality

# Call the function with customized parameters
pp_matrix(
    df_cm,
    cmap="Blues",  # Blues_r (reversed) makes higher values darker
    fz=14,  # Larger font size for better readability
    lw=1.0,  # Thicker lines for better definition
    figsize=[6, 6],
    title="",
    path_to_save_img="publication_confusion_matrix.png"
)