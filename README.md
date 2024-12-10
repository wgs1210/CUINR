1.准备硬件环境<br>
Nvidia GPU ，显存大于10GB即可。<br>

2.准备软硬件环境<br>
(a)conda create -n cuinr python=3.8<br>
(b)conda activate cuinr<br>
(c)pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html<br>
(e)进入CUINR主目录<br>
(f)pip install -r ./cuop.txt<br>

2.预处理训练数据，以Matiwan数据为例<br>
(a)从****上，下载matiwan_uint16.mat并将其放入xx/CUINR/original_data中.<br>
(a)打开文件夹Remote.Sensing-master找到testVCA.m.更改其中的输入文件目录(xx/CUINR/original_data/matiwan_uint16.mat)，执行方法输出预处理数据，并将其放入CUINR指定目录下：xx/CUINR/data/matiwan/xx.mat。<br>
(b)在CUINR目录下，找到main_pavia.py，以及程序入口main，修改rootpath和训练数据目录以及文件名。<br>
(c)开始训练，训练完成后，模型文件会被保存到pathdir/datasetname下<br>
(d)执行main_pavia_eval.py，有提供预训练的模型。执行完成后，xxx.tmp即为最后的压缩文件。计算压缩比，可以用原始文件大小/tmp的文件大小<br>

******请注意，vca提取端元时，文件经过了归一化处理，所以存储后会变大，但这只是属于训练前的预处理，这一步可以在内存中进行，为了方便演示，这里将其存储为mat文件。因此，并不将其作为压缩前的文件，或者在程序结束后，将其删即可。<br>





