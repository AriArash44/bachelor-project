import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

preprocessed = pd.read_csv('./preprocessed.csv')
y = preprocessed["type"]
x= preprocessed.drop(["type"], axis=1)

tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(x)

df_plot = pd.DataFrame()
df_plot['tsne-1'] = X_tsne[:,0]
df_plot['tsne-2'] = X_tsne[:,1]
df_plot['label'] = y.values

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='tsne-1', y='tsne-2',
    hue='label',
    palette='tab10',
    data=df_plot,
    alpha=0.7,
    s=50
)
plt.title('t-SNE visualization of Network dataset (labels)')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
