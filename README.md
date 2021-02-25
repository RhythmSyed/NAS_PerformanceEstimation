# NAS_PerformanceEstimation
Generalized Latency Performance Estimation for Once-For-All Neural Architecture Search

This implementation is adopted from the source code of 
[`Once For All (Cai et al. 2019)`](https://github.com/mit-han-lab/once-for-all) and [`CompOFA (Sahni et al. 2021)`](https://github.com/compofa-blind-review/compofa-iclr21)

We created a jupyter_notebook: [latency_prediction_demo.ipynb](https://github.com/RhythmSyed/NAS_PerformanceEstimation/blob/main/tutorial/latency_prediction_demo.ipynb) which run through
all the required steps to show a working version of the code base.


## Setup
1. conda env create -f environment.yml
2. conda activate latency_predict_env
3. jupyter notebook
4. Open tutorial/latency_prediction_demo.ipynb
5. Run all


## Description of relevant files

1.   latency_prediction_demo.ipynb
    -   Jupyter notebook Demo with steps to run Dataset Creation, Model Evaluation, and OFA NAS
    -   Lists all necessary dependencies for project

2.   latency_predictor_driver.py
    -   Provides code to create latency datasets. Uses imports from evolution_finder to random sample OFA, CompOFA
    -   Code for inference time analysis of latency predictor and measurement

3.   latency_NAS_runner.py
    -   Similar code to demo jupyter notebook. Prepares and run code to perform NAS


4.   evolution_finder.py (modified)
    -   this file was modified to include create_latency_dataset()

5.   latency_predictor/
    6.   datasets/
        -   contains all datasets created for this project, organized into device subfolders. Also includes GPU generalization datasets
    
    7.   model_results/
        -   includes images and text files of all experimental results organized in subfolders
    
    8.   generalized_dataset_combine.py
        -   helper code to add hardware parameters to datasets
    
    9.   Iterations.txt
        -   Log file from search
    
    10.   latency_encoding.py
        -   helper code to perform one hot encoding of child architectures
    
    11.   latency_finetune.py
        -   code for training fine tuned models, case studies, and plotting loss curves
    
    12.   latency_predictor.py
        -   code for creating latency predictors, data_preprocessing, training using RayTune, testing
    
    13.   latency_predictor_generalized.py
        -   code for creating generalized latency predictors
    
    14.   lookup_table_calculation.py
        -   helper code to calculate time taken to create lookup table
    
    15.   other images are results from various experiments


16. checkpoints/
    17.   latency_prediction_model/
        -   contains all trained latency predictors organized in subfolders of hardware and search spaces
