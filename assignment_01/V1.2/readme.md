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
3. 训练保存的模型

TODO FOR PART4:
1. 调整 hyperparameter，training epochs 等

- 5 num_train_epochs 大概需要 30 minutes

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