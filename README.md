# Trianed dataset

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

Other publicly accessible locations of the data:

* https://pan.baidu.com/s/1hjRfATcH30XmdsEMzx1HRg?pwd=hvfl

Data was derived from the following sources:

* Chikusei Dataset: https://naotoyokoya.com/Download.html
* Matiwan Dataset: http://www.hrs-cas.com/a/share/shujuchanpin/
* DFC2018 Houston Dataset: http://dase.grss-ieee.org
* Pavia Centre Dataset: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

# Prepare the software environment
* conda create -n cuinr python=3.8</br>
* conda activate cuinr</br>
* pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html</br>
* cd /xx/CUINR (Enter the CUINR main directory)</br>
* pip install -r ./cuop.txt</br>

# Prepare the training data
打开文件夹Remote.Sensing-master找到testVCA.m.更改其中的输入文件目录(xx/CUINR/original_data)，执行方法输出训练数据，并将其放入CUINR指定目录下。xx/CUINR/data/matiwan/xx.mat。（已matiwan数据集为例）</br>
(b)在CUINR目录下，找到main_pavia.py,以及程序入口main，修改rootpath和训练数据目录以及文件名。</br>
(c)开始训练，训练完成后，模型文件会被保存到pathdir/datasetname下</br>
(d)执行main_pavia_eval.py，有提供预训练的模型。执行完成后，xxx.tmp即为最后的压缩文件，计算压缩比，可以用原始文件大小/tmp的文件</br>

**请注意**
经过vca后，文件经过了归一化处理，所以存储后会变大，但这只是属于训练前的预处理，这一步可以在内存中进行，为了方便演示，这里将其存储为mat文件。因此，并不将其作为压缩前的文件。或者在程序结束后，将其删即可。





