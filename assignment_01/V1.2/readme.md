## Update

基于 A1_pt3_tests.py:

1. A1_pt4_tests.py加入了part4的模型训练；
2. 调整每part的序号，跟 Introduction 大概保持一致
3. A1_pt4_tests.sh提交运行A1_pt4_tests.py的任务
4. 添加用于1~3或part4的测试开关

`log`: 用于提交的运行输出结果

`a1_output*`: 
1. 训练的配置，loss等记录
2. 用于监控/查看训练过程的plot
3. 训练保存的模型: best_model.pt(best model)/model.safetensors(last model)


TODO FOR PART4:
1. 调整 hyperparameter，training epochs 等(done)
2. batch size，MODEL_MAX_LEN 更大

- 5 num_train_epochs 大概需要 30 minutes
- 保留 ≥ 99% 的训练语料词频覆盖（UNK 率 ≤ 1%），同时把嵌入矩阵做得尽量小


## Ouput

**FOR PART1~3:** 

- `log/a1_pt4_tests_84830.out`


**调整过参数的模型：**
**FOR PART4:**

- `a1_output_1.0`: 5 epochs, GRU, hyperparameter 没有调整

- `a1_output_LSTM` & `log/a1_pt4_tests_84985.out`: 早停(6 epochs stop，大概30 minutes)，LSTM（需要在 class A1RNNModel 做调整，未参数化）,hyperparameter 调整

- `a1_output_GRU` & `log/a1_pt4_tests_84987.out`: 早停(9 epochs stop，大概50 minutes)，GRU（需要在 class A1RNNModel 做调整，未参数化）,hyperparameter 调整

`a1_output_LSTM`和`a1_output_GRU`的模型超过100M，完整下载地址：[assignment_01](https://drive.google.com/drive/folders/1xnqq05lLWJM6uTEV6BtxGufgp9sOrjJd?usp=sharing)
## How to run
```
chmod +x ~/ml4nlp/assignment_01/V1.2/sbatch_worker.sh
chmod +x ~/ml4nlp/assignment_01/V1.2/A1_pt4_tests.sh
```

```
sbatch A1_pt4_tests.slurm

# or
./sbatch_worker.sh
```

## Some Useful Command
```
sinfo -N -o "%N %G"

scontrol show node callisto

# login 
srun --nodelist=callisto --pty bash

# show running job
squeue -u USERNAME

# cancel job
scancel JOBID
```