import numpy as np
import matplotlib.pyplot as plt
import deepmimo as dm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

scenario = 'asu_campus_3p5'
dataset = dm.load(scenario)
dataset.compute_channels()

los = dataset.los
positions = dataset.rx_pos[:, :2]

print("Link status breakdown:")
for status, name in [(1, 'LOS (clear)'), (0, 'NLOS (blocked, still reachable)'), (-1, 'Outage (no signal)')]:
    pct = 100 * np.mean(los == status)
    print(f"  {name}: {pct:.1f}%")

X_train, X_test, y_train, y_test = train_test_split(positions, los, test_size=0.2, random_state=0)
clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300, random_state=0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
print(f"\nOverall blockage-state prediction accuracy: {acc:.3f}")
print(classification_report(y_test, pred, target_names=['Outage', 'NLOS', 'LOS']))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, data, title in [(axes[0], y_test, 'True link status'), (axes[1], pred, 'Predicted link status')]:
    sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=data, cmap='RdYlGn', s=4, vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
plt.suptitle('Blockage Prediction: Red = Outage, Yellow = NLOS, Green = Clear LOS')
plt.savefig('blockage_prediction_map.png')
print("Saved blockage_prediction_map.png")
