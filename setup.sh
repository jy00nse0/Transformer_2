#cd /root
#chmod +x setup.sh 
#./setup.sh


cd /root
git clone https://github.com/jy00nse0/Transformer_2.git
conda create -n transformer_py python=3.10 -y
conda activate transformer_py
pip install torch datasets transformers
