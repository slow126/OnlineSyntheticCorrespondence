#!/bin/bash

# Download RGB images
echo "Downloading Virtual KITTI 2 RGB images..."
wget -c https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar

# Download Forward Flow
echo "Downloading Virtual KITTI 2 Forward Flow..."
wget -c https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_forwardFlow.tar

# Download Backward Flow
echo "Downloading Virtual KITTI 2 Backward Flow..."
wget -c https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_backwardFlow.tar

# Download Forward Scene Flow
echo "Downloading Virtual KITTI 2 Forward Scene Flow..."
wget -c https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_forwardSceneFlow.tar

# Download Backward Scene Flow
echo "Downloading Virtual KITTI 2 Backward Scene Flow..."
wget -c https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_backwardSceneFlow.tar

echo "All downloads complete."
