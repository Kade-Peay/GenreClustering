import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import sys

# Get filename from arg
try: 
    filename = sys.argv[1]
except Exception as e:
    print(f"Error with arguments. Usage: python3 plotter.py <filename>")
    exit()

# Read data with error handling
try:
    clustered = pd.read_csv(filename, quotechar='"')
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# Create figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# set color map 
cmap = plt.get_cmap('tab10')

# Create scatter plot
scatter = ax.scatter3D(
    clustered["danceability"],
    clustered["valence"],
    clustered["energy"],
    c=clustered["cluster"],
    cmap=cmap,
    alpha=0.5,
    s=0.5,
    depthshade=True
)

# Labels and title
ax.set_xlabel("Danceability", fontsize=12, labelpad=10)
ax.set_ylabel("Valence", fontsize=12, labelpad=10)
ax.set_zlabel("Energy", fontsize=12, labelpad=10)
plt.title("3D Music Cluster Visualization", fontsize=16, pad=20)

# Colorbar
cbar = plt.colorbar(scatter, pad=0.15)
cbar.set_label('Cluster ID', rotation=270, fontsize=12, labelpad=20)

# Adjust view
ax.view_init(elev=25, azim=45)

# Save and show
outputFilename = "3d_clusters.png"
plt.tight_layout()
plt.savefig(outputFilename, dpi=300, bbox_inches='tight')

# set this off by default as it gave problems. Just open the file after
# plt.show()

print(f"Done creating scatterplot. Saved to {outputFilename}")
