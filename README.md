# Data

The data are available in the 'dataset' folder. Due to file size limitations, only the drug indexes for each fold and a small sample of data are provided to ensure reproducibility. The full dataset and model weights will be released on GitHub at a later time.

# Code

For all the training and inference scripts, first modify the configs file in the 'config' folder, then run 'python xxx.py' from the root folder. 


Prediction model:

'finetune_molt5_prediction.py' is used for training and inference.

ExDDI-S2S:

'finetune_molt5_joint_training.py' is used for training and inference.

ExDDI-MT:

'finetune_molt5_mult_task.py' is used for training and inference.

ExDDI-MTS:

'finetune_molt5_mult_task_inference.py' is used for inference. (Models are trained by ExDDI-MT).

