pip install --upgrade pip

cd ..
git clone https://github.com/moonjongsul/kinect.git
cd kinect
pip install -e .

cd ..
git clone https://github.com/moonjongsul/coord_transform.git
cd coord_transform
pip install -e .

cd ..
cd o3d_3D_recon
pip install -r requirements.txt
pip install -e .
