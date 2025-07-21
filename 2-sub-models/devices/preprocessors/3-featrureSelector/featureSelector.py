import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

normalized_df = pd.read_csv("/run/media/sa/Arash/project/AI model/preprocessors/device/normalized.csv")

X = normalized_df.drop(columns=['label','type_backdoor','type_ddos','type_injection','type_normal','type_password','type_ransomware','type_scanning','type_xss'])
y = normalized_df[['label','type_backdoor','type_ddos','type_injection','type_normal','type_password','type_ransomware','type_scanning','type_xss']]

corr_matrix = X.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

threshold = 0.95

high_corr_features = []
for col in upper.columns:
    for row in upper.index:
        if upper.loc[row, col] > threshold:
            high_corr_features.append((row, col, upper.loc[row, col]))

print("\nHighly Correlated Feature Pairs (Threshold > {:.2f}):".format(threshold))
for feature1, feature2, correlation_value in high_corr_features:
    print(f"{feature1} ↔ {feature2}: Correlation = {correlation_value:.4f}")

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print("\nHighly correlated features to drop:", to_drop)

pipeline = Pipeline([
    ('dropper', ColumnDropper(columns_to_drop=to_drop)),
    ('pca', PCA(n_components=0.9, svd_solver='full'))
])

X_train_transformed = pipeline.fit_transform(X)

pca_feature_names = [f"PC{i+1}" for i in range(X_train_transformed.shape[1])]
X_train_pca_df = pd.DataFrame(X_train_transformed, columns=pca_feature_names)

scaler = MinMaxScaler()
X_pca_scaled = scaler.fit_transform(X_train_pca_df)
X_pca_scaled_df = pd.DataFrame(X_pca_scaled, columns=X_train_pca_df.columns)

final_train_df = pd.concat([X_pca_scaled_df, y.reset_index(drop=True)], axis=1)
print("\nFinal training DataFrame shape:", final_train_df.shape)

print("preprocess dataset tasks completed!")
final_train_df.to_csv("devices.csv", index=False)

cumulative_explained = np.cumsum(pipeline.named_steps['pca'].explained_variance_ratio_)
plt.plot(np.arange(1, len(cumulative_explained) + 1), cumulative_explained, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.savefig('pca_explained_variance.png')
print("\nPlot saved as 'pca_explained_variance.png'.")

joblib.dump(pipeline, 'normalized_to_full_preprocessed_pipeline.pkl')
print("\nPreprocessing pipeline saved.")