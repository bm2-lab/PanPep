# PAML
## Introduction 
A framework constructed in three levels for predicting the peptide and TCR binding recoginition.
## Requirements  
* python == 3.9.7  
* pytorch == 1.10.2  
* numpy == 1.21.2  
* pandas == 1.4.1  
* scipy == 1.7.3  
#### * Note : you should install CUDA and cuDNN version compatible with the pytorch version [Version Searching](https://www.tensorflow.org/install/source). 
## Usage  

    Usage: PAML.py [options]
    Required:
          --learning_setting STRING: choosing the learning setting: few-shot, zero-shot and majority
          --input STRING: the path to the input data file (*.csv)
          --output STRING: the path to the output data file (*.csv)

    Optional:
          --update_lr FLOAT: task-level inner update learning rate (default: 0.01)
          --update_step_test INT: update steps for finetunning (default: 3)
          --C INT: Number of bases (default: 3)
          --R INT: Peptide Index matrix vector length (default: 3)
          --L INT: Peptide embedding length (default: 75) 
We provided three examples in different learning settings to show you how to use PAML to predict the peptide and TCR regconition. 
### Few-shot learning setting 
Command:  

    python PAML.py --learning_setting few-shot --input ./Data/Example_few-shot.csv --output ./Output/Example_few-shot_output.csv
### Zero-shot learning setting 
Command:  

    python PAML.py --learning_setting zero-shot --input ./Data/Example_zero-shot.csv --output ./Output/Example_zero-shot_output.csv
### Majority learning setting 
Command: 

    python PAML.py --learning_setting majority --update_step_test 1000 --input ./Data/Example_majority.csv --output ./Output/Example_majority_output.csv
## Citation
## Contacts
2011398@tongji.edu.cn  
1810994@tongji.edu.cn  
qiliu@tongji.edu.cn
