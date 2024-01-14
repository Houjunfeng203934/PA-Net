1. python "your path/PA-Net/nnunet/dataset_conversion/Task777_mutul.py"
#  Data preparation

2. python "your path/PA-Net/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py" -t 777
#  Data preprocessing

3. python "your path/PA-Net/nnunet/run/run_training.py" 3d_fullres DUNetTrainer 777 4
#  Training

4. python "your path/PA-Net/nnunet/inference/predict_simple.py" -i "your input path" -o "your output path" -t 777 -f 4 -tr DUNetTrainer
#  test
