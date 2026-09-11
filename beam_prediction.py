import numpy as np
import matplotlib.pyplot as plt
import deepmimo as dm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- 1. Load the scenario ---
scenario = 'asu_campus_3p5'
dataset = dm.load(scenario)
dataset.compute_channels()

# --- 2. Build a codebook of 16 candidate beams ---
panel_size = (8, 1)  # matches the 8 tx antennas in this scenario
angles = np.linspace(-60, 60, 16)
codebook = np.array([dm.steering_vec(panel_size, phi=a).squeeze() for a in angles])  # (16, 8)

# --- 3. Ground-truth optimal beam per user ---
h = dataset.channels[:, 0, :, 0]                  # (num_users, 8)
recv_power = np.abs(h @ codebook.conj().T) ** 2   # (num_users, 16)
labels = np.argmax(recv_power, axis=1)

# --- 4. Train a model: position -> best beam ---
X = dataset.rx_pos[:, :2]   # user (x, y) location
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)

clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300, random_state=0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# --- 5. Metrics vs baselines ---
acc = accuracy_score(y_test, pred)

proba = clf.predict_proba(X_test)
top3 = np.argsort(proba, axis=1)[:, -3:]
top3_acc = np.mean(np.any(top3 == y_test[:, None], axis=1))

random_acc = 1 / len(angles)
majority_acc = np.max(np.bincount(y_test)) / len(y_test)

print(f'Model accuracy:    {acc:.3f}')
print(f'Top-3 accuracy:    {top3_acc:.3f}')
print(f'Random baseline:   {random_acc:.3f}')
print(f'Majority baseline: {majority_acc:.3f}')

# --- 6. Results bar chart ---
plt.figure(figsize=(5, 4))
plt.bar(['Random', 'Majority', 'Model (ours)'], [random_acc, majority_acc, acc],
        color=['gray', 'gray', 'steelblue'])
plt.ylabel('Accuracy')
plt.title('Beam Prediction Accuracy vs Baselines')
plt.savefig('beam_prediction_results.png')
print('Saved chart to beam_prediction_results.png')

# --- 7. Ground truth vs predicted beam map ---
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='tab20', s=4)
axes[0].set_title('True best beam')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')

axes[1].scatter(X_test[:, 0], X_test[:, 1], c=pred, cmap='tab20', s=4)
axes[1].set_title('Model predicted beam')
axes[1].set_xlabel('x (m)')

plt.suptitle('Beam Assignment Across Campus: Ground Truth vs Model')
plt.savefig('beam_prediction_map.png')
print('Saved map to beam_prediction_map.png')
