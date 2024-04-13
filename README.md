Miniproject for CS-GY 6923 S24 Deep Learning \
To run the code, download the repo and run ```sbatch slurm_script.s``` with slurm or \
```python3 main.py --model ResNetWide --epochs 200 --lr 0.1 --batch_size 128 --optimizer SGD --data_augmentation True``` \
To generate Kaggle labels, run the notebook kaggle_test.ipynb \
All the results are saved under the 'saved_results' and 'new_saved_results' directory, \
the test accuracy of each trail and parameter configs are stored under the config.json file of each trail.
