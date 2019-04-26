# CheXNet implementation in PyTorch

Reproduce the [CheXNet](https://arxiv.org/abs/1711.05225) algorithm for pathology detection in 
frontal chest X-ray images. This implementation is based on approach presented [here](https://github.com/zoogzog/chexnet). Ten-crops 
technique is used to transform images at the testing stage to get better accuracy. 

The highest accuracy evaluated with AUROC was 0.8508 (with original data set,see the model m-25012018-123527 in the models directory).
The same training (70%), validation (10%) and testing (20%) datasets were used as in [this](https://github.com/arnoweng/CheXNet) 
implementation.


Result for all models show in **result.pdf**.

Many thanks to [zoogzog](https://github.com/zoogzog/chexnet) for detailed implematation of tranmsformatian dataset and auroc comoutation.

![alt text](test/heatmap.png)

## Prerequisites
* Python 3.6
* Pytorch==0.3.1
* 2 GPU(recommend NVIDIA TESLA P100 or equivalent)
* OpenCV (for generating CAMs)
* Linux OS(Google cloud platform,debian 9,8CPU(56G RAM)

## Usage
* Download the ChestX-ray14 database from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)
* Unpack archives in separate directories (e.g. images_001.tar.gz into images_001)
* The original dataset contains over 100,000 images,which takes too long to train a model.Thus,we downsize the data with the  policy on :drop duplicate where patient id and label are the same from [Data_Entry_2017.csv
](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345).The smaller dataset contains almost half of         original(train_1.txt,test_1.txt,val_1.txt are the original split txt index for immages,while train.txt,test.txt,val.txt are  the final splitting txt file to train our model,which took much less time).
  
* Run **python Main.py** to run test using the pre-trained model (m-25012018-123527 for DenseNet121)
* Use the **runTrain()** function in the **Main.py** to train a model from scratch

This implementation allows to conduct experiments with 3 different densenet architectures: densenet-121, densenet-169,ResNet50,Se_ResNet50,Se_DenseNet121.

* To generate CAM of a test file run script HeatmapGenerator 

## Results
The highest accuracy 0.8508 was achieved by the model m-25012018-123527 for DenseNet121 started by pretrained model(see the models directory).

| Pathology     | AUROC         |
| ------------- |:-------------:|
|    Mean       | 0.8508        |
| Atelectasis   | 0.8321        |
| Cardiomegaly  | 0.9107        |
| Effusion      | 0.8860        |
| Infiltration  | 0.7145        |
| Mass          | 0.8653        |
| Nodule        | 0.8037        |
| Pneumonia     | 0.7655        |
| Pneumothorax  | 0.8857        |
| Consolidation | 0.8157        |
| Edema         | 0.9017        |
| Emphysema     | 0.9422        |
| Fibrosis      | 0.8523        |
| P.T.          | 0.7948        |
| Hernia        | 0.9416        |

## Computation time
The training was done using double Tesla P100 GPU and 6h on downsize data.

## Reference paper
[CheXNet](https://arxiv.org/abs/1711.05225),[Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf),
[DenseNet](https://arxiv.org/pdf/1608.06993.pdf) and [ResNet](https://arxiv.org/pdf/1512.03385.pdf).


