import numpy as np
import matplotlib.pyplot as plt
import deepmimo as dm
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scenario = 'asu_campus_3p5'
dataset = dm.load(scenario)
dataset.compute_channels()

h = dataset.channels[:, 0, :, 0]
X = np.hstack([h.real, h.imag])
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

# Scale to a reasonable range so the neural network can actually train
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def nmse(true, pred):
    return np.mean(np.sum((true - pred) ** 2, axis=1)) / np.mean(np.sum(true ** 2, axis=1))

compression_dim = 4

pca = PCA(n_components=compression_dim).fit(X_train)
X_test_pca = pca.inverse_transform(pca.transform(X_test))
nmse_pca = nmse(X_test, X_test_pca)

autoencoder = MLPRegressor(hidden_layer_sizes=(32, compression_dim, 32),
                            max_iter=500, random_state=0)
autoencoder.fit(X_train, X_train)
X_test_ae = autoencoder.predict(X_test)
nmse_ae = nmse(X_test, X_test_ae)

print(f"Compression ratio: 16:{compression_dim} ({16/compression_dim:.1f}x)")
print(f"PCA reconstruction NMSE:         {nmse_pca:.4f}")
print(f"Autoencoder reconstruction NMSE: {nmse_ae:.4f}")
improvement = 100 * (nmse_pca - nmse_ae) / nmse_pca
print(f"Improvement over PCA: {improvement:.1f}%")

plt.figure(figsize=(5, 4))
plt.bar(['PCA', 'Autoencoder (ours)'], [nmse_pca, nmse_ae], color=['gray', 'steelblue'])
plt.ylabel('NMSE (lower is better)')
plt.title(f'Channel Compression at {16/compression_dim:.0f}x Ratio')
plt.savefig('compression_results.png')
print("Saved compression_results.png")
