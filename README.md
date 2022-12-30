# Neural_Networks_from_Scratch
Neural Networks From Scratch - Using only Numpy

### [Paper: The Core Mathematics of a Neural Network](https://github.com/ShlomoStept/Neural_Networks_from_Scratch/blob/main/NN_From_Scratch_The%20Core%20Mathematics.pdf) | [Video](https://www.youtube.com/watch?v=w8yWXqWQYmU) | [Data](https://www.kaggle.com/competitions/digit-recognizer/data?select=train.csv) 

![](nn_fs.gif)

## Get started
You can set up the environment with all dependencies like so:
```
conda create --name NN-FS python=3.8
conda activate NN-FS
pip install numpy pandas
```

## High-Level structure
* Paper: Explores in more detail the core mathematics of the neural network.
* data: MNIST Dataset of handwritten digits
* model: Simple 3-Layer Neural Network

## How to Run


1. Please download the python or jupyter-notebook file 

2. Dowload the dataset - train.csv from (https://www.kaggle.com/competitions/digit-recognizer/data?select=train.csv), and place into a folder named mnist_data
  
    - At this point your project structure should look like : 
```
├── Neural_Network_from_Scratch.ipynb
├── Neural_Network_from_Scratch.py
└── mnist_data
    └── train.csv
```

3. Then either run the cells of jupyter notebook or run the python file using 
  ```
  python Neural_Network_from_Scratch.py
  ```

## Disclaimer(s)

1. The result produced by this code might be slightly different when running on a different GPU. 

