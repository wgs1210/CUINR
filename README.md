# Training data

[https://doi.org/10.5281/zenodo.14391084](https://doi.org/10.5281/zenodo.14391084)

## Description of the data and file structure

The uploaded data contains two ZIP files: one is the original hyperspectral image (original_data.zip), and the other is the hyperspectral image processed by VCA (data.zip).

The data used for training is from data.zip, which is the pre-processed file. For convenience in demonstration, it is stored as a MAT file.
### Files and variables

#### File: original\_data.zip

**Description:** The uncompressed file.

#### File: data.zip

**Description:** The pre-processed file.
## Access information

Data was derived from the following sources:

* Chikusei Dataset: https://naotoyokoya.com/Download.html
* Matiwan Dataset: https://aistudio.baidu.com/datasetdetail/100218
* DFC2018 Houston Dataset: http://dase.grss-ieee.org
* Pavia Centre, and Cuprite Dataset: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
* Indian Pines Dataset: https://purr.purdue.edu/publications/1947/1
* Jasper Ridge Dataset: https://lesun.weebly.com/hyperspectral-data-set.html


# Descriptions and implementations of key algorithms
CUINR uses E-NeRV as the backbone network, specifically detailed in /xx/CUINR/model/E_NeRV_pavia.py. The unmixing reconstruction module and the two physical constraints, ANC and ASC, are implemented in the AutoEncoder and SumToOneLoss within E_NeRV_pavia.py. The alternative implementations of Lasc and Clamp can be found in the softmax function within AutoEncoder.<br>

# Prepare the software environment
* conda create -n cuinr python=3.8</br>
* conda activate cuinr</br>
* pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html</br>
* cd /xx/CUINR (Enter the CUINR main directory)</br>
* pip install -r ./cuop.txt</br>

# Prepare the training data
* Open the folder Remote.Sensing-master and locate the file testVCA.m. 
Modify the input file directory (/xx/CUINR/original_data/matiwan_uint16.mat), execute the method, and output the preprocessed data. 
Then place the resulting file in the designated CUINR directory: /xx/CUINR/data/matiwan/matiwan_vca.mat. (using the Matiwan dataset as an example). Alternatively, you can directly use matiwan_vca.mat from data.zip. </br>
* Locate the CUINR main directory, find the file main_pavia.py and the program entry point main. Modify the rootpath and the training data directory /xx/CUINR/data/matiwan as well as the file name matiwan_vca.mat.</br>
# Train
* Run ./train.bash. After training is complete, the model files will be saved in /xx/CUINR/pathdir/datasetname.</br>
# Eval
* run python  main_pavia_eval.py. The execution process is similar to train. This file contains a demo for testing the pretrained model.<br>
### Please note: 
After VCA, the file undergoes normalization, which causes its size to increase after storage(data.zip). However, this is just part of the preprocessing before training, and this step can be done in memory. For the sake of demonstration, it is stored as a .mat file here. Therefore, it should not be considered as the file before compression. Alternatively, you can delete it after the program finishes.





