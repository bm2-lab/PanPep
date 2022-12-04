# PanPep
## Introduction 
PanPep is a framework constructed in three levels for predicting the peptide and TCR binding recognition. We have provided the trained basic meta learner and external memory and users can choose different settings based on their data available scenerio:  
* Few known TCRs for a peptide: few-shot setting 
* No known TCRs for a peptide: zero-shot setting
* A plenty of known TCRs for a peptide: majority setting 
 
![Figure 1](https://user-images.githubusercontent.com/89248357/185394001-acb797db-b8de-43a1-8d33-f770165921dd.png)

## Requirements  
* python == 3.9.7  
* pytorch == 1.10.2  
* numpy == 1.21.2  
* pandas == 1.4.1  
* scipy == 1.7.3  
#### * Note : you should install CUDA and cuDNN version compatible with the pytorch version [Version Searching](https://pytorch.org/). 
## Usage  

    Usage: PanPep.py [options]
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
We provided three examples in different learning settings to show you how to use PanPep to predict the peptide and TCR recognition. 
### Few-shot setting 
Command:  

    python PanPep.py --learning_setting few-shot --input ./Data/Example_few-shot.csv --output ./Output/Example_few-shot_output.csv 
    
* input.csv: input *.csv file contains three columns: Peptide, CDR3 and Label, which represents the peptide sequence, TCR CDR3 squence and their binding specificity.
In the Label column, there are three values: 1 indicating binding, 0 indicating non-binding and unknown. Then, known peptide-CDR3 pairs will be used to construct the TCR support set to fine-tune the basic meta learner and unknown peptide-CDR3 pairs will be used to construct the TCR query set for being predicted.
* output.csv: out *.csv file contains three columns: Peptide, CDR3 and Score, which represents the peptide sequence, TCR CDR3 squence and their predicted binding score. All the peptide-CDR3 pairs are the unknown pairs in the input file.
### Zero-shot setting 
Command:  

    python PanPep.py --learning_setting zero-shot --input ./Data/Example_zero-shot.csv --output ./Output/Example_zero-shot_output.csv 
    
* input.csv: input *.csv file contains two columns: Peptide and CDR3, which represents the peptide sequence, TCR CDR3 squence.
* output.csv: out *.csv file contains three columns: Peptide, CDR3 and Score, which represents the peptide sequence, TCR CDR3 squence and their predicted binding score. All the peptide-CDR3 pairs are the pairs in the input file.

### Majority setting 
Command: 

    python PanPep.py --learning_setting majority --update_step_test 1000 --input ./Data/Example_majority.csv --output ./Output/Example_majority_output.csv 

* update_step_test: 1000 represents the basic meta learner will be fine-tuned 1000 times for each peptide-level task and then be used to predict the binding score of TCRs in its TCR query set.
* input.csv: input *.csv file contains three columns: Peptide, CDR3 and Label, which represents the peptide sequence, TCR CDR3 squence and their binding specificity.
In the Label column, there are three values: 1 indicating binding, 0 indicating non-binding and unknown. Then, known peptide-CDR3 pairs will be used to construct the TCR support set to fine-tune the basic meta learner and unknown peptide-CDR3 pairs will be used to construct the TCR query set for being predicted.
* output.csv: out *.csv file contains three columns: Peptide, CDR3 and Score, which represents the peptide sequence, TCR CDR3 squence and their predicted binding score. All the peptide-CDR3 pairs are the unknown pairs in the input file.
## Citation
## Contacts
bm2-lab@tongji.edu.cn
