#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
python setup.py install --user

# Compactness Constraint
cd $HOME/extensions/expansion_penalty
python setup.py install --user

# knn cuda
cd $HOME/extensions/KNN_CUDA
python setup.py install --user

# Pointnet2
cd $HOME/extensions/Pointnet2/pointnet2
python setup.py install --user

