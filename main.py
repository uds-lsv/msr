import argparse
from load_utils import prepare_data
from mws_utils import create_logger, save_args, create_trainer, save_config
import numpy as np
import torch
import random
import wandb
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='trec', choices=['Yoruba', 'Hausa',
                                                                        'trec', 'agnews', 'imdb', 'yelp',
                                                                        'mitr', 'conll', 'ontonotes'])
    parser.add_argument('--data_root', type=str, default='/local/user_name_aabbcc/data/wrench/tmp_data')
    parser.add_argument('--log_root', type=str, default="/local/user_name_aabbcc/outputs/tmp/mws")
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--task_type', type=str, default='text_cls',
                        choices=['text_cls', 'ner'])
    parser.add_argument('--trainer_name', type=str, default='vanilla',
                        choices=['vanilla', 'msr', 'msr_ner'])
    parser.add_argument('--model_name', type=str, default='roberta-base',
                        choices=['bert-base-multilingual-cased', 'roberta-base'])

    parser.add_argument('--pooling_strategy', type=str, default='pooler_output',
                        choices=['pooler_output', 'mean', 'max'])
    parser.add_argument('--store_model', type=int, default=0, help='store model after training,'
                                                                   'set to 0 to disable')
    parser.add_argument('--metric', type=str, default='accuracy', choices=['accuracy', 'f1_macro'])


    # pre-processing
    parser.add_argument('--truncate_mode', type=str, default='last',
                        choices=['hybrid, last'], help='last: last 510 tokens, hybrid: first 128 + last 382')
    parser.add_argument('--max_sen_len', type=int, default=128)
    parser.add_argument('--special_token_offsets', type=int, default=2,
                        help='number of special tokens used in bert tokenizer for text classification')
    parser.add_argument('--include_val_in_train', action='store_true',
                        help='if we include validation set in the training set, '
                             'imdb dataset will have 25000 training instances')

    # BERT settings
    parser.add_argument('--bert_dropout_rate', type=float, default=0.1)
    parser.add_argument('--re_init_pooler', action='store_true')


    # training
    parser.add_argument('--num_training_steps', type=int, default=10)
    parser.add_argument('--train_eval_freq', type=int, default=10)
    parser.add_argument('--val_eval_freq', type=int, default=50)
    parser.add_argument('--eval_freq', type=int, default=50, help='one batch is one step, '
                                                                  'eval_freq=n means eval at every n step')

    parser.add_argument('--nl_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=50)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--use_clean', action='store_true',
                        help='use the ground truth label in training,'
                             'default: do NOT (!) to use')



    parser.add_argument('--fast_eval', action='store_true',
                        help='use 10% of the test set for evaluation, to speed up evaluation by approximation')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='freeze the bert backbone, i.e. use bert as feature extractor')
    parser.add_argument('--return_entity_level_metrics', action='store_true')

    # COSINE trainer
    parser.add_argument('--T2', type=int, default=100)
    parser.add_argument('--T3', type=int, default=50)
    parser.add_argument('--self_training_power', type=float, default=2, help='power of pred score')
    parser.add_argument('--self_training_contrastive_weight', type=float, default=1)
    parser.add_argument('--self_training_eps', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--self_training_confreg', type=float, default=0.1)
    parser.add_argument('--cosine_teacher_label_type', type=str, default='soft', choices=['soft', 'hard'])
    parser.add_argument('--cosine_distmetric', type=str, default="l2", choices=['cos', 'l2'],
                        help='distance type. Choices = [cos, l2]')




    # MSR Trainer
    parser.add_argument('--meta_teacher_lr', type=float, default=2e-5)
    parser.add_argument('--v_split', type=int, default=1)
    parser.add_argument('--teacher_warmup_steps', type=int, default=0)
    parser.add_argument('--consistency_lambda', type=float, default=1.0)
    parser.add_argument('--add_weights_to_meta', type=float, default=0.0)
    parser.add_argument('--add_weights_to_train', type=float, default=0.0)
    parser.add_argument('--conf_check', action='store_true', help='check some statistics MSR')

    # Standard Self-Training
    parser.add_argument('--st_conf_threshold', type=float, default=0.9)
    parser.add_argument('--self_training_iteration', type=int, default=2)
    parser.add_argument('--student_training_steps', type=int, default=10)
    parser.add_argument('--teacher_init_weights_dir', type=str, default='')


    # Optimizer
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--exp_decay_rate', type=float, default=0.9998)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--discr', action='store_true',
                        help='different learning rate for different layers')
    parser.add_argument('--layer_learning_rate', type=str, nargs='+', default=[2e-5])
    parser.add_argument('--layer_learning_rate_decay', type=float, default=0.95)

    # Hardware and Reproducibility
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cuda_device', type=str, default="0")
    parser.add_argument('--manualSeed', type=int, default=1234, help='random seed for reproducibility')
    parser.add_argument('--noisy_label_seed', type=int, default=1234, help='random seed for reproducibility')

    args = parser.parse_args()

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.backends.cudnn.benchmark = False
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.deterministic = True

    wandb_log_name = f"tws-{args.trainer_name}" if args.exp_name=='' else args.exp_name
    wand_dir = "./wandb_logs"
    if not os.path.exists(wand_dir):
        os.mkdir(wand_dir)
    wandb.init(
        project=wandb_log_name,
        dir=wand_dir,
        config={})
    wandb.config.update(args)

    # Create a Handler for logging records/messages to a file
    logger, log_dir = create_logger(args.log_root, args)
    save_args(log_dir, args)
    logger.info("Training started")
    print(f'log dir: {log_dir}')

    # sanity checks
    if args.dataset in ['Yoruba', 'Hausa', 'trec', 'agnews', 'imdb', 'yelp']:
        assert args.task_type == 'text_cls'
    elif args.dataset in ['mitr', 'conll', 'ontonotes']:
        assert args.task_type == 'ner'
    else:
        raise NotImplementedError("[main.py]: Unknown dataset type")

    if args.use_clean:
        assert args.trainer_name == 'vanilla', "Only the vanilla model may work on clean labels"

    logger.info(f'loading {args.dataset}')
    r_state = np.random.RandomState(args.noisy_label_seed)
    full_dataset = prepare_data(args, logger, r_state)

    trainer = create_trainer(args, logger, log_dir, r_state)
    trainer.train(args, logger, full_dataset)
    save_config(log_dir, 'exp_config', args)  # model_config could be updated during model creation
    logger.info(f"Logs located at {log_dir}")

if __name__=='__main__':
    main()