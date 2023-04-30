# Meta Self-Refinement
Meta Self-Refinement (MSR) is a noise-robust learning framework that enables learning with unreliable labels.

Please refer to our paper for a detailed description of MSR:
[Meta Self-Refinement for Robust Learning with Weak Supervision (EACL 2023)](https://arxiv.org/abs/2205.07290)



# Data Preparation
Download the [WRENCH datasets](https://github.com/JieyuZ2/wrench) to `/path/to/data`. 
The  directory structure should look like the following:
```
/path/to/data
│
└───trec
│   │   train.json
│   │   test.json
│   │   valid.json
│   │   readme.md
│   │   label.json
|
└───imdb
│   │   train.json
│   │   test.json
│   │   valid.json
│   │   readme.md
│   │   label.json
...
```


# Run Tasks
### Example 1:
Fine-tune PLM directly with weak labels on IMDB:
1. specify the log root `LOG_ROOT=/path/to/log` and  data root `DATA_ROOT=/path/to/data`
2. Run the following script to fine-tune the RoBERTa model (vanilla model, no noise-handling).
```
CUDA_VISIBLE_DEVICES=$CUDA_ID python3 ../main.py \
--dataset imdb \
--log_root $LOG_ROOT \
--data_root $DATA_ROOT \
--trainer_name vanilla \
--model_name roberta-base \
--nl_batch_size 32 \
--eval_batch_size 32 \
--max_sen_len 64 \
--lr 2e-5 \
--num_training_steps 600 \
--eval_freq 25 \
--train_eval_freq 25 \
--manualSeed 1234
```

### Example 2:
Fine-tune PLM with weak labels and apply MSR on IMDB:
1. specify the log root `LOG_ROOT=/path/to/log` and  data root `DATA_ROOT=/path/to/data`.
2. MSR requires an initial teacher model `MODEL_HOME=/path/to/models`. You can first train a vanilla model to have the initial teacher model for MSR.
3. run the following script to train MSR.

```
CUDA_VISIBLE_DEVICES=$CUDA_ID python3 ../main.py \
--dataset imdb \
--log_root $LOG_ROOT \
--data_root $DATA_ROOT \
--trainer_name msr \
--model_name roberta-base \
--nl_batch_size 16 \
--eval_batch_size 32 \
--max_sen_len 64 \
--lr 2e-5 \
--meta_teacher_lr $meta_teacher_lr \
--teacher_warmup_steps 2000 \
--add_weights_to_train 1.0 \
--st_conf_threshold $st_conf_threshold \
--conf_check \
--num_training_steps 5000 \
--teacher_init_weights_dir "${MODEL_HOME}/wrench/imdb_vanilla" \
--eval_freq 25 \
--train_eval_freq 25 \
--manualSeed 1234
```

# Remarks
- The higher library can be downloaded from [here](https://github.com/facebookresearch/higher). Please clone the library and install from the source. Installation using pip does not support the AdamW optimizer used in this code.
- The two datasets in low-resource African languages used in our experiments are from [here](https://github.com/uds-lsv/transfer-distant-transformer-african). The `preprocess_hausa.py` and `preprocess_yoruba.py` can be used to obtain the WRENCH format.



# Citation

Please cite our paper if you find this code useful for your research:
```
@inproceedings{zhu-etal-2023-meta,
    title = "Meta Self-Refinement for Robust Learning with Weak Supervision",
    author = "Zhu, Dawei  and
      Shen, Xiaoyu  and
      Hedderich, Michael  and
      Klakow, Dietrich",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.74",
    pages = "1043--1058"
}

```