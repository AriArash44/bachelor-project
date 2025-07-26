import pandas as pd
from sklearn.cluster import KMeans

INPUT_CSV = "../1-VAE/vae_latents_z.csv" 
N_CLUSTERS = 32
OUTPUT_CSV = "kmeans_centers_32.csv"

def main():
    df = pd.read_csv(INPUT_CSV)
    X  = df.values.astype("float32")
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    centers_df = pd.DataFrame(centers, columns=df.columns)
    centers_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved {N_CLUSTERS} cluster centers to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
