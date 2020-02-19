# 文本匹配



## Setup

This code uses Python 3.6+ and Tensorflow-GPU 1.6. Pytorch 1.0 Clone the repository and install all required packages. It is recommended to use the [Anaconda package manager](https://www.anaconda.com/download/#macos). After installing Anaconda -

```shell
conda create --name dstc8 python=3.6
source activate dstc7
pip install -r requirements.txt
```



## 数据

数据来源于DSTC8 Noesis II - Predicting Responses Track：[训练数据](http://ibm.biz/dstc8_track2_data)和[测试数据](http://ibm.biz/dstc8_track2_test_data)



#### Prepare the data

Before training, the data needs to converted into a suitable format. The script `convert_dstc8_data.py` can be used to convert data for both advising and ubuntu datasets. 将数据放到RE2_pt/orig/ubuntu目录下



```shell
cd RE2_pt/data

python convert_dstc8_data.py --train_in orig/ubuntu/task-1.ubuntu.train.json
--train_out ubuntu/train.txt
--dev_in orig/ubuntu/task-1.ubuntu.dev.json
--dev_out ubuntu/dev.txt
--answers_file ubuntu/ubuntu_task_1_candidates.txt
--save_vocab_path ubuntu/ubuntu_task_1_vocab.txt
```



## Usage

To train a new text matching model, run the following command: 

```bash
cd RE2_pt

python train.py $config_file.json5
```



