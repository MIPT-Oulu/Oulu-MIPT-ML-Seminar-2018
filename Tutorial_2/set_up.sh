# Loading the modules
module load python-env/3.5.3-ml

# Installing the requirements
pip3.5 install -r requirements.txt --user
export PATH=/homeappl/home/atiulpin/.local/bin/:$PATH

# Setting things up
mkdir $HOME/.kaggle/
echo "{\"username\":\"$1\",\"key\":\"$2\"}" > $HOME/.kaggle/kaggle.json

# Downloading the data
kaggle competitions download -c invasive-species-monitoring --force
mkdir data
mv $HOME/.kaggle/competitions/invasive-species-monitoring/* data/