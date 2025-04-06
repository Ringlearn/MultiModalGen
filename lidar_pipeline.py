import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LiDARPipeline(nn.Module):
    def __init__(self, voxel_size=[0.2, 0.2, 0.2], coors_range=[0, -40, -3, 70, 40, 5], max_points=35, max_voxels=20000):
        super(LiDARPipeline, self).__init__()

        # Parameters
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.coors_range = np.array(coors_range, dtype=np.float32)
        self.max_points = max_points
        self.max_voxels = max_voxels

        # Encoder components
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(256, 512)  # Embedding dimension: 512

        # Decoder components
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 43440 * 3)  # Output point cloud dimensions (x, y, z)

    def load_kitti_bin_file(self, file_path):
        """Load a KITTI .bin file containing LiDAR point cloud data."""
        point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)  # (N, 4) -> x, y, z, intensity
        return point_cloud[:, :3]  # Use only x, y, z coordinates

    def points_to_voxel(self, points):
        # Voxelization logic
        voxelmap_shape = (self.coors_range[3:] - self.coors_range[:3]) / self.voxel_size
        voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())

        num_points_per_voxel = np.zeros(shape=(self.max_voxels,), dtype=np.int32)
        coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
        voxels = np.zeros(shape=(self.max_voxels, self.max_points, points.shape[-1]), dtype=points.dtype)
        coors = np.zeros(shape=(self.max_voxels, 3), dtype=np.int32)

        voxel_num = min(len(points), self.max_voxels)  # Simulating voxelization
        for i in range(voxel_num):
            coors[i] = np.random.randint(0, 80, size=(3,))
            num_points_per_voxel[i] = min(self.max_points, points.shape[0])
            voxels[i, :num_points_per_voxel[i], :] = points[:num_points_per_voxel[i]]

        return voxels[:voxel_num], coors[:voxel_num], num_points_per_voxel[:voxel_num]

    def encode(self, x):
        # Encoding process
        x = x.permute(0, 3, 1, 2)  # Shape -> (Batch, Channels, Voxel Count, Max Points)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # Shape: (batch_size, latent_dim, voxel_count, max_points)
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # Shape: (batch_size, latent_dim)
        x = self.projection(x)  # Shape: (batch_size, embedding_dim)
        return x

    def decode(self, x):
        # Decoding process
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Shape: (batch_size, num_points * 3)
        x = x.view(x.shape[0], 43440, 3)  # Shape: (batch_size, num_points, 3)
        return x

    def forward(self, file_path):
        # Load point cloud from KITTI .bin file
        point_cloud = self.load_kitti_bin_file(file_path)
        print(f"Loaded point cloud shape: {point_cloud.shape}")

        # Voxelize the point cloud
        voxels, _, _ = self.points_to_voxel(point_cloud)  # Voxelize the point cloud
        voxel_data = torch.tensor(voxels).unsqueeze(0)  # Shape: (Batch, M, Max Points, Features)

        # Encode to latent embeddings
        embeddings = self.encode(voxel_data)  # Encode to latent embeddings

        # Decode back to point cloud
        reconstructed_point_cloud = self.decode(embeddings)  # Decode back to point cloud

        return embeddings, reconstructed_point_cloud


# Instantiate and Run the Unified Pipeline
pipeline = LiDARPipeline()
bin_file_path = "n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801005947122.pcd.bin"

# Run the pipeline
embeddings, reconstructed_point_cloud = pipeline(bin_file_path)

# Print Shapes
print("Embedded Feature Shape:", embeddings.shape)  # Expected: (batch_size, 512)
print("Reconstructed Point Cloud Shape:", reconstructed_point_cloud.shape)  # Expected: (batch_size, 43440, 3)