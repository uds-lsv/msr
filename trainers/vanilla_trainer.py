import wandb
import torch
import os
from torch.utils.data import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from models import AdaptiveCrossEntropy
from trainers.trainer import Trainer
from trainers.early_stopper import EarlyStopper
from tqdm import tqdm


class VanillaTrainer(Trainer):
    def __init__(self, args, logger, log_dir, random_state):
        super(VanillaTrainer, self).__init__(args, logger, log_dir, random_state)
        self.store_model_flag = True if args.store_model == 1 else False
        self.use_clean = args.use_clean

    def get_batch(self, d_loader, d_iter):
        try:
            d_batch = next(d_iter)
        except StopIteration:
            d_iter = iter(d_loader)
            d_batch = next(d_iter)

        return d_batch, d_iter

    def train(self, args, logger, full_dataset):
        assert args.gradient_accumulation_steps <= 1, "this trainer does not support gradient accumulation for now"
        logger.info('Bert Vanilla Trainer: training started')
        device = self.device
        l_set, ul_set, val_set, test_set = self.process_dataset(full_dataset)
        l_loader = torch.utils.data.DataLoader(l_set, batch_size=args.nl_batch_size, shuffle=True, num_workers=0)
        t_loader = torch.utils.data.DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=True, num_workers=0)
        assert val_set is not None, 'We need a validation set'
        v_loader = torch.utils.data.DataLoader(val_set, batch_size=args.eval_batch_size,
                                               shuffle=False,
                                               num_workers=0)

        tr_loader = torch.utils.data.DataLoader(l_set, batch_size=args.nl_batch_size, shuffle=True, num_workers=0)
        tr_iter = iter(tr_loader)

        model = self.create_model(args)
        model = model.to(device)
        num_training_steps = args.num_training_steps
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(args, model)

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        optimizer_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                              num_training_steps=num_training_steps)


        global_step = 0
        if self.store_model_flag:
            early_stopper_save_dir = os.path.join(self.log_dir, 'early_stopper_model')
            if not os.path.exists(early_stopper_save_dir):
                os.makedirs(early_stopper_save_dir)
        else:
            early_stopper_save_dir = None

        early_stopper = EarlyStopper(patience=args.patience, delta=0,
                                     large_is_better=self.es_large_is_better,
                                     save_dir=early_stopper_save_dir, verbose=False,
                                     trace_func=logger.info)
        # ce_loss_fn = nn.CrossEntropyLoss()
        ce_loss_fn = AdaptiveCrossEntropy(args=args, num_classes=self.num_classes, reduction='mean')

        # make sure that the number of epochs covers the steps needed
        num_epochs = (num_training_steps // (len(l_loader) + 1)) + 1
        wandb.run.summary["num_epochs"] = num_epochs

        best_val_acc = -1

        # train the network
        for step in tqdm(range(num_training_steps), desc=f'training steps', ncols=150):

            tr_batch, tr_iter = self.get_batch(tr_loader, tr_iter)
            nl_input_ids = tr_batch['input_ids'].to(device)
            nl_attention_mask = tr_batch['attention_mask'].to(device)

            if args.use_clean:
                nl_labels = tr_batch['c_labels'].to(device)
            else:
                nl_labels = tr_batch['n_labels'].to(device)

            bs = len(tr_batch['n_labels'])

            model.train()
            model.zero_grad()
            outputs = model(nl_input_ids, nl_attention_mask)['logits']
            loss = ce_loss_fn(outputs, nl_labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer_scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            batch_ce_loss = loss.item()

            if self.needs_eval(args, global_step):
                test_res = self.eval_model(args, logger, device, ce_loss_fn, t_loader, model, fast_mode=args.fast_eval,
                                           verbose=False)
                val_res = self.eval_model(args, logger, device, ce_loss_fn, v_loader, model, fast_mode=args.fast_eval,
                                          verbose=False)

                self.log_score_to_wandb(args, test_res, global_step, tag="test")
                self.log_score_to_wandb(args, val_res, global_step, tag="validation")

                val_score = self.get_val_score(val_res)  # track validation loss or accuracy, or F-1?
                early_stopper.register(val_score,
                                       model,
                                       optimizer)
            if global_step == num_training_steps or early_stopper.early_stop:
                break


        if self.store_model_flag:
            self.save_model(logger, model, 'last_model_weights.bin')

        model = self.create_model(args)
        model.load_state_dict(early_stopper.get_final_res()['es_best_model'])
        model = model.to(device)
        test_res = self.eval_model(args, logger, device, ce_loss_fn, t_loader, model, verbose=False)
        self.summary_best_score_to_wandb(args, test_res, tag='best')
