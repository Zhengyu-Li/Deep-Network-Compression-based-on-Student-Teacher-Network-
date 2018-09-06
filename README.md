# Deep-Network-Compression-based-on-Student-Teacher-Network-
# Introduction
This guideline is to guide you to train/test the original DenseNet model or the Student-Teacher model. The paper can be found in XXX. 
All of our codes are based on Tensorflow and Python.
The file organization is shown below.  
**teacher_student**  
----cifar100  
--------cifar-100-python  
------------train(cifar100 training data)  
------------test(cifar100 test data)  
----data_providers  
--------cifar.py(for original training data)  
--------base_provider.py(for original training data)  
--------cifar2.py(for intermediate outputs)  
--------data_providers2.py(for intermediate outputs)  
----models  
--------dense_net.py  
----eval.py (for model evaluation)  
----get_teacher.py (for get intermediate outputs)  
----run_dense_net.py (for training/test student network)  
----student1.py  
----student2.py  
----student3.py  
**multi-gpu**  
----cifar100  
--------train  
--------test  
----architectures  
--------densene.py  
----arch.py  
----data_loader.py(read data)  
----train.py  
----eval.py  
***  
# Training & Testing Original DenseNet
Here are the paper and official implementation of DenseNet:
Paper: [DenseNet]
Offical Implementation: https://github.com/liuzhuang13/DenseNet
In our implementation, we use multiple GPUs to train DenseNet-BC (L=190, k=40), and single GPU to train DenseNet-BC (L=100, k=12). The details are shown below.
### Use Single GPU
Our codes (teacher_student folder) are based on [Illarion Khlestov's implementation]

**How to Run**
- First, install the requirements in "requirements.txt" in "teacher_student" folder.
- Start a new train. The model will be saved in "saves" folder. The logs will be saved in "log" folder. Example run: 
  ```sh
  $ python run_dense_net.py --train \
                          --dataset=C100org \
                          --epoch=300 \
                          --growth_rate=12 \
                          --depth=100 
  ```
- Continue training with previous model. The command is same as train a new model. You need to confirm the dataset name, growth rate and depth are same as what you set before.
- Test model. Example run:
  ```sh
  python run_dense_net.py --test \
                          --dataset=C100org \
                          --epoch=300 \
                          --growth_rate=12 \
                          --depth=100 
  ```
- For more details, please follow the comments in python files.
### Use multiple GPUs
Our codes (multi-gpu folder) are based on [Arashno's implementation]

**How to Run**
- First, install the requirements in "requirements.txt" in "multi-gpu" folder.  
- Start a new train. Example run:
  ```sh
  $ CUDA_VISIBLE_DEVICES=1,2 python train.py --architecture densenet \
                                             --data_info=./dense/cifar100/train \
                                             --num_epoch=100 \
                                             --num_gpu=3 \
                                             --batch_size=64
  ```
  CUDA_VISIBLE_DEVICES is to set GPUs you want to use. Please change the numbers according to your machine.
- Continue training with previous model. Example run:
  ```sh
  $ CUDA_VISIBLE_DEVICES=1,2 python train.py --architecture densenet \
                                             --data_info=./cifar100/train \
                                             --num_epoch=100 \
                                             --num_gpu=3 \
                                             --batch_size=64 \
                                             --retrain_from=./multi-gpu/dense-190/
  ```
- Test model. Example run:
  ```sh
  $ python eval.py --architecture=densenet \
                   --data_info=./cifar100/test \
                   --batch_size=200
  ```
- For more details, please follow the comments in python files.
***  
# Training & Testing Student DenseNet
The basic idea is to use intermediate outputs of the teacher network to guide the student network. For more details about the Student-Teacher Network and our training strategies, please reference to my thesis [Deep Network Compression based on Student-Teacher Network].
Our codes are bassed on [Illarion Khlestov's implementation].
#### Get intermediate outputs from Teacher
Before training the student network, you need to save the intermediate outputs from teacher for convenience. The main codes are in "get_teacher.py" file.
**Usage:**
```sh 
$ python get_teacher.py
```
It will generate "train_data.h5" (contain the original images, labels and all of the intermediate outputs) and "test_data.h5" files.
#### Train & Test Student
In our project, we use three training strategies: same learning rate without extended layer (student1.py, Section 3.3.1), same learning rate with extended layer (student2.py, Section 3.3.2) and different learning rates(student3.py, Section 3.3.3).
**1. Use same learning rate**
- First, install the requirements in "requirements.txt" in "teacher_student" folder.
- Second, follow the comments and change configurations in file ```./teacher_student/data_providers/cifar2.py```. 
- Third, make sure the network structure and training configurations in file ```student1.py``` or ```student2.py``` are right.
- Run ```student1.py``` or ```student2.py```
- Run ```eval.py``` to evaluate model (follow the comments in file to change setting)
**2. Use different learning rate**
- First, install the requirements in "requirements.txt" in "teacher_student" folder.
- Second, follow the comments and change configurations in file ```./teacher_student/data_providers/cifar2.py```. 
- Third, make sure the network structure and training configurations in file ```student3.py``` are right.
- Run ```student3.py```
- Run ```eval.py``` to evaluate model (follow the comments in file to change setting)
***  
# Use TensorBoard on Macbook
Tensorboard is a convenient tool to check training. The details can be found at https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard. 
#### Installation
It will install tensorboard automatically when installing tensorflow.
#### When training on same laptop
When training model on same laptop, you just need to input this command in your terminal, and set the directory as which you save the summaries.
```sh
$ tensorboard --logdir=path/to/log-directory
```
If you face the error like
```locale.Error: unsupported locale setting```,
you need to set your encoding format to "English, UTF-8":
```sh
$ export LANGUAGE=en_US.UTF-8
$ export LANG=en_US.UTF-8
$ export LC_ALL=en_US.UTF-8
```
Then run TensorBoardd again.
Once the TensorBoard has be opened sucessfully, you will see 
```sh
TensorBoard 1.5.1 at http://127.0.0.1:6006
```
Now you can open TensorBoard at http://127.0.0.1:6006. (recommand Chrome)
#### When training on server
When training on the server, the summary files will be saved on the server, too. You need to transfer the port 6006 of the remote server into the port 16006 (or other free port) of your machine. Use below command to connect to the server:
```sh
$ ssh -L 16006:127.0.0.1:6006 user@server-address
```
Then start TensorBoard in server with same command.
Now you can open TensorBoard at http://127.0.0.1:16006 in your machine's browser.


[Arashno's implementation]: <https://github.com/arashno/tensorflow_multigpu_imagenet> 
[Illarion Khlestov's implementation]: <https://github.com/ikhlestov/vision_networks>  
[DenseNet]: <https://arxiv.org/abs/1608.06993>
[Deep Network Compression based on Student-Teacher Network]: <http:172.0.0.1>
