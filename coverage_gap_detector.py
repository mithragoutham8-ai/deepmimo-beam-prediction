import numpy as np
import matplotlib.pyplot as plt
import deepmimo as dm
from sklearn.cluster import KMeans

scenario = 'asu_campus_3p5'
dataset = dm.load(scenario)
dataset.compute_channels()

threshold_dbw = -110
power = dataset.power[:, 0]  # strongest path per user
positions = dataset.rx_pos[:, :2]
dead_mask = power < threshold_dbw

print(f"Dead zone users: {dead_mask.sum()} / {len(power)} "
      f"({100*dead_mask.sum()/len(power):.1f}%)")

dead_points = positions[dead_mask]
K = 3
kmeans = KMeans(n_clusters=K, random_state=0, n_init=10).fit(dead_points)
new_sites = kmeans.cluster_centers_
print("Proposed new small-cell locations (x, y):")
print(new_sites)

plt.figure(figsize=(7, 6))
sc = plt.scatter(positions[:, 0], positions[:, 1], c=power, cmap='viridis', s=3)
plt.colorbar(sc, label='Power (dBW)')
plt.scatter(dead_points[:, 0], dead_points[:, 1], c='red', s=4, alpha=0.3, label='Dead zone')
plt.scatter(new_sites[:, 0], new_sites[:, 1], c='yellow', edgecolor='black',
            s=250, marker='*', label='Proposed new small cell')
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Coverage Gaps and Proposed Small-Cell Placement')
plt.savefig('coverage_gap_map.png')
print("Saved coverage_gap_map.png")
