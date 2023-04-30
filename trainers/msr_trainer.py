import os
from pathlib import Path
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
from trainers.trainer import Trainer
from trainers.early_stopper import EarlyStopper
import higher
from transformers import logging as t_logging
from transformers import get_linear_schedule_with_warmup
from models import AdaptiveCrossEntropy


class MSRTrainer(Trainer):
    def __init__(self, args, logger, log_dir, random_state):
        super(MSRTrainer, self).__init__(args, logger, log_dir, random_state)
        self.store_model_flag = True if args.store_model == 1 else False
        self.add_weights_to_meta = True if args.add_weights_to_meta == 1.0 else False
        self.add_weights_to_train = True if args.add_weights_to_train == 1.0 else False
        self.st_conf_threshold = args.st_conf_threshold
        t_logging.set_verbosity_error()

    def soft_cross_entropy_loss(self, y_logit, y_soft_label, reduction='mean'):
        # TODO: CE or KLDiv here?
        kldiv_loss_no_reduction = -torch.sum(F.log_softmax(y_logit, dim=1) * y_soft_label, dim=1)
        if reduction == 'mean':
            Lx = torch.mean(kldiv_loss_no_reduction)
        elif reduction == 'none':
            Lx = kldiv_loss_no_reduction
        else:
            raise ValueError("soft_cross_entropy_loss: unknown reduction type")
        return Lx

    def soft_cross_entropy_loss_no_reduction(self, y_logit, y_soft_label):
        kldiv_loss_no_reduction = -torch.sum(F.log_softmax(y_logit, dim=1) * y_soft_label, dim=1)
        return kldiv_loss_no_reduction

    def soft_frequency(self, logits, probs=False, soft=True):
        """
        Unsupervised Deep Embedding for Clustering Analysis
        https://arxiv.org/abs/1511.06335
        """
        power = self.args.self_training_power
        if not probs:
            softmax = nn.Softmax(dim=1)
            y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
        else:
            y = logits
        f = torch.sum(y, dim=0)
        t = y ** power / f
        # print('t', t)
        t = t + 1e-10
        p = t / torch.sum(t, dim=-1, keepdim=True)
        return p if soft else torch.argmax(p, dim=1)

    def get_batch(self, d_loader, d_iter):
        try:
            d_batch = next(d_iter)
        except StopIteration:
            d_iter = iter(d_loader)
            d_batch = next(d_iter)

        return d_batch, d_iter


    def train(self, args, logger, full_dataset):
        assert args.gradient_accumulation_steps <= 1, "this trainer does not support gradient accumulation for now"
        logger.info('Bert MetaCleaning Trainer: training started')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        l_set, ul_set, val_set, test_set = self.process_dataset(full_dataset)

        if ul_set is not None:
            tr_set_full = self.concat_datasets(l_set, ul_set)
        else:
            tr_set_full = l_set

        wandb.run.summary["train size"] = len(tr_set_full)
        wandb.run.summary["val size"] = len(val_set)
        wandb.run.summary["test size"] = len(test_set)

        tr_loader = torch.utils.data.DataLoader(tr_set_full, batch_size=args.nl_batch_size, shuffle=True, num_workers=0)
        tr_iter = iter(tr_loader)

        t_loader = torch.utils.data.DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=True, num_workers=0)
        assert val_set is not None, 'The MSR Algorithm needs a validation set'

        # Step 2: we need to load the previous best model.
        teacher_weights_dir = args.teacher_init_weights_dir
        teacher_weights_path = Path(teacher_weights_dir) / "model_dict.pt"
        teacher_model = self.create_model(args)
        teacher_model.load_state_dict(torch.load(teacher_weights_path))
        teacher_model = teacher_model.to(device)

        eval_ce_loss_fn = AdaptiveCrossEntropy(args=args, num_classes=self.num_classes, reduction='mean')
        initial_teacher_test_res = self.eval_model(args, logger, device, eval_ce_loss_fn,
                                                   t_loader, teacher_model, fast_mode=False, verbose=False)
        wandb.run.summary["init_teacher_acc"] = initial_teacher_test_res['score_dict']['accuracy']

        wandb.log({f"t test acc": initial_teacher_test_res['score_dict']['accuracy'],
                   f"t test macro-f1": initial_teacher_test_res['score_dict']['macro avg']["f1-score"]},
                  step=0)

        student_model = self.create_model(args)
        student_model = student_model.to(device)

        num_training_steps = args.num_training_steps

        teacher_optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(args, teacher_model)
        teacher_model_optimizer = optim.AdamW(teacher_optimizer_grouped_parameters, lr=args.meta_teacher_lr)
        teacher_optimizer_scheduler = get_linear_schedule_with_warmup(teacher_model_optimizer,
                                                                      num_warmup_steps=args.teacher_warmup_steps,
                                                                      num_training_steps=num_training_steps)

        student_optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(args, student_model)
        student_model_optimizer = optim.AdamW(student_optimizer_grouped_parameters, lr=args.lr)

        global_step = 0
        val_loss_fn = torch.nn.CrossEntropyLoss()

        if self.store_model_flag:
            early_stopper_save_dir = os.path.join(self.log_dir, 'early_stopper_model')
            if not os.path.exists(early_stopper_save_dir):
                os.makedirs(early_stopper_save_dir)
        else:
            early_stopper_save_dir = None

        early_stopper = EarlyStopper(patience=200, delta=0, save_dir=early_stopper_save_dir, verbose=False,
                                     trace_func=logger.info)

        v_split = args.v_split
        v_loader = torch.utils.data.DataLoader(val_set, batch_size=args.eval_batch_size // v_split,
                                               shuffle=True, num_workers=0)
        v_iter = iter(v_loader)

        # train the network
        for step in range(num_training_steps):

            tr_batch, tr_iter = self.get_batch(tr_loader, tr_iter)
            nl_input_ids = tr_batch['input_ids'].to(device)
            nl_attention_mask = tr_batch['attention_mask'].to(device)
            c_labels = tr_batch['c_labels'].to(device)

            meta_net = student_model
            meta_optimizer = student_model_optimizer

            meta_net.zero_grad()
            teacher_model.zero_grad()
            meta_net.train()
            teacher_model.train()

            meta_loss = 0
            for i in range(v_split):
                v_batch, v_iter = self.get_batch(v_loader, v_iter)
                v_input_ids = v_batch['input_ids'].to(device)
                v_attention_mask = v_batch['attention_mask'].to(device)
                v_clean_labels = v_batch['c_labels'].to(device)

                with higher.innerloop_ctx(meta_net, meta_optimizer) as (meta_model, meta_diffopt):
                    teacher_logits = teacher_model(nl_input_ids, nl_attention_mask)['logits']
                    teacher_soft_labels = F.softmax(teacher_logits, dim=1)
                    student_logits = meta_model(nl_input_ids, nl_attention_mask)['logits']

                    if self.add_weights_to_meta:
                        teacher_pseudo_target = teacher_soft_labels.detach()
                        weight = torch.sum(-torch.log(teacher_pseudo_target + 1e-6) * teacher_pseudo_target, dim=1)  # Entropy
                        weight = 1 - weight / np.log(weight.size(-1))
                        w = (weight > self.st_conf_threshold)
                        meta_loss = torch.mean(w * self.soft_cross_entropy_loss(student_logits,
                                                                                teacher_soft_labels,
                                                                                reduction="none"))
                    else:
                        meta_loss = self.soft_cross_entropy_loss(student_logits, teacher_soft_labels)

                    meta_diffopt.step(meta_loss)

                    meta_pred_val = meta_model(v_input_ids, v_attention_mask)['logits']
                    l_g_meta = val_loss_fn(meta_pred_val, v_clean_labels) / v_split

                    grad_of_grads = torch.autograd.grad(outputs=l_g_meta, inputs=teacher_model.parameters())

                    meta_loss += l_g_meta.item()

                for p, g in zip(teacher_model.parameters(), grad_of_grads):
                    if p.grad is None:
                        p.grad = g
                    else:
                        p.grad += g
                del grad_of_grads


            wandb.log({f"meta_loss": meta_loss},
                      step=global_step)
            teacher_model_optimizer.step()
            teacher_optimizer_scheduler.step()
            teacher_model.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher_model(nl_input_ids, nl_attention_mask)['logits']
                teacher_soft_labels = F.softmax(teacher_logits, dim=1)



            student_model.zero_grad()
            student_model.train()
            student_logits = student_model(nl_input_ids, nl_attention_mask)['logits']


            if self.add_weights_to_train:
                teacher_pseudo_target = teacher_soft_labels.detach()
                weight = torch.sum(-torch.log(teacher_pseudo_target + 1e-6) * teacher_pseudo_target, dim=1)  # Entropy
                weight = 1 - weight / np.log(weight.size(-1))
                w = (weight > self.st_conf_threshold)
                student_ce_loss = torch.mean(w * self.soft_cross_entropy_loss(student_logits, teacher_soft_labels.detach()))
            else:
                student_ce_loss = self.soft_cross_entropy_loss(student_logits, teacher_soft_labels.detach())
            student_ce_loss.backward()
            student_model_optimizer.step()
            student_model.zero_grad()

            global_step += 1
            wandb.log({"student_ce_loss": student_ce_loss.item()}, step=global_step)

            if global_step % args.train_eval_freq == 0 and global_step != 0:
                logger.info(f'Step: {global_step}, student_loss: {student_ce_loss.item()}')

            if self.needs_eval(args, global_step):
                test_res = self.eval_model(args, logger, device, eval_ce_loss_fn,
                                           t_loader, student_model, fast_mode=args.fast_eval,
                                           verbose=False)
                val_res = self.eval_model(args, logger, device, eval_ce_loss_fn,
                                          v_loader, student_model, fast_mode=args.fast_eval,
                                          verbose=False)

                self.log_score_to_wandb(args, test_res, global_step, tag="st test")
                self.log_score_to_wandb(args, test_res, global_step, tag="st val")


                val_score = self.get_val_score(val_res) # track validation loss or accuracy, or F-1?
                early_stopper.register(val_score,
                                       student_model,
                                       student_model_optimizer)

                test_res = self.eval_model(args, logger, device, eval_ce_loss_fn, t_loader, teacher_model, fast_mode=args.fast_eval,
                                           verbose=False)

                val_res = self.eval_model(args, logger, device, eval_ce_loss_fn, v_loader, teacher_model, fast_mode=args.fast_eval,
                                          verbose=False)

                self.log_score_to_wandb(args, test_res, global_step, tag="teacher test")
                self.log_score_to_wandb(args, val_res, global_step, tag="teacher val")



            if global_step == num_training_steps or early_stopper.early_stop:
                break

        student_model = self.create_model(args)
        student_weights = early_stopper.get_final_res()["es_best_model"]
        student_model.load_state_dict(student_weights)
        student_model = student_model.to(device)

        test_res = self.eval_model(args, logger, device, eval_ce_loss_fn, t_loader, student_model, fast_mode=args.fast_eval,
                                   verbose=False)

        self.summary_best_score_to_wandb(args, test_res, tag='best student')

        if self.store_model_flag:
            self.save_model(logger, student_model, 'model_weights.bin')

        return {"global_step": global_step, "best_student_model": student_model}
