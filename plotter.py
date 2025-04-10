import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

# Set style for better visuals
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Before clustering 
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
df = pd.read_csv("tracks_features.csv")
df.columns = ["id","name","album","album_id","artists","artist_ids","track_number",
              "disc_number","explicit","danceability","energy","key","loudness",
              "mode","speechiness","acousticness","instrumentalness","liveness",
              "valence","tempo","duration_ms","time_signature","year","release_date"]
sns.scatterplot(x=df["danceability"], y=df["energy"], hue=df["key"], 
                palette="viridis", alpha=0.6)
plt.title("Original: Danceability vs Energy (Colored by Key)")
plt.xlabel("Danceability")
plt.ylabel("Energy")

# After clustering 
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
clustered_df = pd.read_csv("output.csv")
sns.scatterplot(x="danceability", y="energy", hue="cluster", 
                data=clustered_df, palette="hls", alpha=0.6)
plt.title("Clustered: Danceability vs Energy")
plt.xlabel("Danceability")
plt.ylabel("Energy")

# Adjust layout and save
plt.tight_layout()
plt.savefig("scatterplot.png", dpi=300, bbox_inches='tight')
plt.show()