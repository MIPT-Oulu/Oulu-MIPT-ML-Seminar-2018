pip install kaggle
pip install torchvision
pip install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl --user
pip install opencv-python
pip install tqdm
pip install scikit-learn

conda install -c bioconda p7zip

# Setting things up
mkdir $HOME/.kaggle/
echo "{\"username\":\"$1\",\"key\":\"$2\"}" > $HOME/.kaggle/kaggle.json
chmod 600 $HOME/.kaggle/kaggle.json

# Downloading the data
kaggle competitions download -c invasive-species-monitoring --file train.7z
kaggle competitions download -c invasive-species-monitoring --file train_labels.csv.zip

mkdir data
mv $HOME/.kaggle/competitions/invasive-species-monitoring/* data/

cd data/
7z x train.7z
rm train.7z
unzip train_labels.csv.zip
