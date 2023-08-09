## Training and Testing
### 1. Pre-training

In order to get better performance, we train the teacher model firstly, then we use the pre-trained weights to initialize
the teacher branch in the person search model.

The teacher model can be regarded as a simple person Re-ID model, so we need to construct the training and test 
data of person Re-ID style from the existing person search datasets.

We cropped out parts of each person's body from the CUHK-SYSU and G2APS datasets and reorganized them 
in the form of Market1501, a dataset commonly used for person Re-ID. Since the images of the PRW dataset 
and Market1501 dataset come from the same group of raw video, there is no need to reorganize it. 

It should be emphasized that when we transform the person search dataset into the person Re-ID dataset, 
we strictly abide by the training set and test set division of the original dataset.

You can reconstruct it yourself, or download my reconstructed dataset from here.

You can then pretrain the two teacher models on three datasets using the commands in [./teacher_coat/running_cmd](./teacher_coat/running_cmd) 
and [./teacher_seqnet/running_cmd](./teacher_seqnet/running_cmd). Of course, you can skip this step, we have prepared the [model weights(code 1357)](https://pan.baidu.com/s/16VycrlBtPnqmULhgojo0Jw) for you.

Note: Parameters such as CONFIG DATA_ROOT OUTPUT_DIR in each command need to be re-specified according to the actual situation.

### 2. Training

If you want to train the SeqNet+HKD model, you need to prepare the teacher branch pre-training weights obtained in 
step 1 and place them in the [SeqNet-HKD/teacherckpt](SeqNet-HKD/teacherckpt) . Then go to [SeqNet-HKD/running_cmd](SeqNet-HKD/running_cmd), select instructions 
based on the dataset you are using, and train your model.
For COAT+HKD, after entering the COAT-HKD, the operation is similar to the above.

Note: Parameters such as CONFIG DATA_ROOT OUTPUT_DIR in each command need to be re-specified according to the actual situation.

### 3.Testing
If you want to test with the weights of our trained person search model, let's use SeqNet+HKD as an example to demonstrate. 
First of all, download the weights of our trained HKD model here, put them in the [SeqNet-HKD/evalckpt](SeqNet-HKD/evalckpt) directory, 
and then select a test command in the [SeqNet-HKD/running_cmd](SeqNet-HKD/running_cmd) according to the dataset you use to test.

In addition, you can replace the weight parameters in the test command with your own trained model weights.

Note: Parameters such as CONFIG DATA_ROOT OUTPUT_DIR in each command need to be re-specified according to the actual situation.