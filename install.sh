pip install --upgrade pip

cd ..
git clone https://github.com/moonjongsul/kinect.git
cd kinect
pip install -e .

cd ..
git clone https://github.com/moonjongsul/coord_transform.git
cd coord_transform
pip install -e .

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
pip install -U git+https://github.com/UnstructuredWork/udp.git
pip install git+https://github.com/cocodataset/panopticapi.git
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet

cd ..
cd o3d_3D_recon
pip install -r requirements.txt
pip install -e .

pip install rospkg