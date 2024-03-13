conda create -n gen3d python=3.10
conda activate gen3d
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

sudo apt-get install graphviz
# mkdir -p out/tune