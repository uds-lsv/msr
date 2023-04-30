import torch
import os
import numpy as np
from trainers.majority_voting import MajorityVoting
import wandb
from mws_dataset import TextBertDataset
from models import TextBert, RobertaForTokenClassification
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as pr_score
from datasets import load_metric
from transformers import DataCollatorForTokenClassification
from trainers.early_stopper import EarlyStopper

class Trainer:
    def __init__(self, args, logger, log_dir, random_state):
        self.args = args
        self.logger = logger
        self.log_dir = log_dir
        self.label_txt_list = None
        self.l2id = None
        self.id2l = None
        self.num_classes = None
        self.r_state = random_state
        self.store_model_flag = True if args.store_model == 1 else False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_best_val_acc_save_path = os.path.join(self.log_dir, 'best_val_acc_model.pt')
        self.best_val_optimizer_save_path = os.path.join(self.log_dir, 'best_val_acc_opt.pt')
        self.collator = None

        if args.task_type == 'ner':
            self.eval_fn = self.ner_eval
            self.es_large_is_better = True
        elif args.task_type == 'text_cls':
            self.eval_fn = self.tc_eval
            self.es_large_is_better = False
        else:
            raise ValueError("[Trainer]: unknown task_type")


    def get_val_score(self, val_score_dict):
        if self.args.task_type == 'ner':
            return val_score_dict['score_dict']['overall_f1']
        elif self.args.task_type == 'text_cls':
            return val_score_dict['loss']
        else:
            raise ValueError("[Trainer]: unknown task_type")


    def io_id_to_bio_id(self, a):
        bio_ids = []
        last_io = -1
        for i in a:
            if i== -100: # subtoken id, skip
                bio_ids.append(-100)
                continue
            if i == 0:
                bio_ids.append(0)
            else:
                if i == last_io:
                    bio_ids.append(int(i * 2))  # to I
                else:
                    bio_ids.append(int(i * 2 - 1))  # to B
            last_io = i
        return bio_ids


    def ner_eval(self, predictions, labels):
        metric = load_metric("seqeval")

        # predictions = np.argmax(predictions, axis=2)
        predictions = [self.io_id_to_bio_id(p) for p in predictions]
        labels = [self.io_id_to_bio_id(l) for l in labels]

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2l[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2l[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if self.args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "overall_f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    def tc_eval(self, predictions, labels):
        classification_score_dict = classification_report(labels,
                                                          np.array(predictions).flatten(),
                                                          target_names=self.label_txt_list,
                                                          output_dict=True)
        return classification_score_dict


    def create_model(self, args):
        bert_config = {'num_classes': self.num_classes}
        if args.task_type == 'ner':
            model = RobertaForTokenClassification(args, None, **bert_config)
        elif args.task_type == 'text_cls':
            model = TextBert(args, None, **bert_config)
        return model

    def process_dataset(self, full_dataset):
        train_set = full_dataset["train_set"]
        val_set = full_dataset["validation_set"]
        test_set = full_dataset["test_set"]
        self.l2id = full_dataset["l2id"]
        self.id2l = full_dataset["id2l"]

        if self.args.task_type == 'ner':
            num_classes = len(self.l2id.keys())
            assert num_classes % 2 !=0, "number of BIO classes should always be odd"
            self.num_classes = int((num_classes+1)/2)
            # self.collator = DataCollatorForTokenClassification(train_set.tokenizer, pad_to_multiple_of= None)
        elif self.args.task_type == 'text_cls':
            self.num_classes = len(self.l2id.keys())
        else:
            raise ValueError("[Trainer]: Unknown task_type")


        if train_set.n_labels == []:
            self.logger.info("creating noisy labels using majority vote")
            self.majority_voter = MajorityVoting()
            n_labels = self.majority_voter.predict(train_set, **{"task_type": self.args.task_type})
            train_set.n_labels = n_labels

        train_set.gen_bert_input()
        l_set, ul_set = train_set.get_covered_subset()

        n_labels = self.majority_voter.predict(val_set, **{"task_type": self.args.task_type})
        val_set.n_labels = n_labels
        n_labels = self.majority_voter.predict(test_set, **{"task_type": self.args.task_type})
        test_set.n_labels = n_labels

        noisy_label_train_stat = self.eval_fn(l_set.n_labels, l_set.labels)
        noisy_label_test_stat = self.eval_fn(test_set.n_labels, test_set.labels)

        if self.args.task_type == 'ner':
            self.logger.info(f"Majority Voting - F1 on training set: {noisy_label_train_stat['overall_f1']}")
            self.logger.info(f"Majority Voting - F1 on test set: {noisy_label_test_stat['overall_f1']}")
        elif self.args.task_type == 'text_cls':
            self.logger.info(f"Majority Voting - Accuracy on training set: {noisy_label_train_stat['accuracy']}")
            self.logger.info(f"Majority Voting - Accuracy on test set: {noisy_label_test_stat['accuracy']}")
        else:
            raise ValueError("[Trainer]: Unknown task type")

        val_set.gen_bert_input()
        test_set.gen_bert_input()

        self.logger.info(f"weakly labeled samples: {len(l_set)}")
        self.logger.info(f"unlabeled samples: {len(ul_set)}") if ul_set is not None else None
        self.logger.info(f"validation samples: {len(val_set)}")
        self.logger.info(f"test samples: {len(test_set)}")

        return l_set, ul_set, val_set, test_set



    def save_model(self, logger, model, model_name):
        output_path = os.path.join(self.log_dir, model_name)
        torch.save(model.state_dict(), output_path)
        logger.info(f"model saved at: {output_path}")


    def train(self, args, logger, full_dataset):
        raise NotImplementedError()

    def needs_eval(self, args, global_step):
        if global_step % args.eval_freq == 0 and global_step != 0:
            return True
        else:
            return False


    def eval_model(self, args, logger, device, loss_fn,
                   eval_set_loader, model,
                   on_noisy_labels=False, fast_mode=False, verbose=False,  **kwargs):
        all_preds = []
        all_y = []
        model.eval()
        loss_sum = 0.0

        num_batches = len(eval_set_loader)/10 if fast_mode else 0

        with torch.no_grad():
            for idx, eval_batch in enumerate(eval_set_loader):
                input_ids = eval_batch['input_ids'].to(device)
                attention_mask = eval_batch['attention_mask'].to(device)
                if on_noisy_labels:
                    targets = eval_batch['n_labels'].to(device)
                else:
                    targets = eval_batch['c_labels'].to(device)

                y_logits = model(input_ids, attention_mask)['logits']
                loss = loss_fn(y_logits, targets)
                loss_sum += loss.item()
                y_preds = torch.max(y_logits, -1)[1].cpu()
                all_preds.extend(y_preds.numpy())
                all_y.extend(list(targets.cpu()))

                if fast_mode and idx > num_batches:
                    break

            score_dict = self.eval_fn(all_preds, all_y)

        return {'score_dict': score_dict,
                'loss': loss_sum/(len(all_y))}

    def log_score_to_wandb(self, args, result, global_step, tag):
        if result is None:
            return

        if args.task_type == 'ner':
            wandb.log({f"{tag} f1": result['score_dict']['overall_f1']},
                      step=global_step)
        elif args.task_type == 'text_cls':
            wandb.log({f"{tag} acc": result['score_dict']['accuracy'],
                       f"{tag} macro-f1": result['score_dict']['macro avg']["f1-score"]},
                      step=global_step)
        else:
            raise ValueError("[Trainer]: Unknown task type")



    def summary_best_score_to_wandb(self, args, res_dict, tag):
        if args.task_type == 'ner':
            self.logger.info(f"{tag} F1: {res_dict['score_dict']['overall_f1']}")
            wandb.run.summary[f"{tag} score"] = res_dict['score_dict']['overall_f1']
        elif args.task_type == 'text_cls':
            self.logger.info(f"{tag} Accuracy: {res_dict['score_dict']['accuracy']}")
            wandb.run.summary[f"{tag} score"] = res_dict['score_dict']['accuracy']
        else:
            raise ValueError("[Trainer]: Unknown task type")


    def concat_datasets(self, set1, set2):
        assert set1.id2l == set2.id2l
        combined_dataset = TextBertDataset(self.args, input_data=None, tokenizer=set1.tokenizer, id2l=set1.id2l)
        combined_dataset.ids = set1.ids + set2.ids
        combined_dataset.labels = set1.labels + set2.labels
        combined_dataset.examples = set1.examples + set2.examples
        combined_dataset.weak_labels = set1.weak_labels + set2.weak_labels
        combined_dataset.n_labels = set1.n_labels + set2.n_labels
        combined_dataset.bert_input = {k: torch.cat((v, set2.bert_input[k]), dim=0) for k, v in set1.bert_input.items()}
        self.logger.info(f"Datasets concatenated, size after concatenation: {len(combined_dataset)}")

        return combined_dataset


    def get_optimizer_grouped_parameters(self, args, model):
        # no_decay = ['bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'gamma', 'beta']
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']
        if args.discr:
            if len(args.layer_learning_rate) > 1:
                groups = [(f'layer.{i}.', args.layer_learning_rate[i]) for i in range(12)]
            else:
                lr = args.layer_learning_rate[0]
                assert lr == args.lr
                groups = [(f'layer.{i}.', lr * pow(args.layer_learning_rate_decay, 11 - i)) for i in range(12)]
            group_all_attn_layers = [f'layer.{i}.' for i in range(12)]
            no_decay_optimizer_parameters = []
            decay_optimizer_parameters = []

            # set learning rate for self-attention layers, 12 layers for bert-base
            for g, l in groups:
                decay_optimizer_parameters.append(
                    {
                        'params': [p for n, p in model.named_parameters() if
                                   not any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                        'weight_decay': args.weight_decay, 'lr': l
                    }
                )
                no_decay_optimizer_parameters.append(
                    {
                        'params': [p for n, p in model.named_parameters() if
                                   any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                        'weight_decay': 0.0, 'lr': l
                    }
                )
            # set learning rate for anything that don't belong to the attention layers, e.g. embedding layer,
            # the dense layer
            group_all_parameters = [
                {'params': [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all_attn_layers)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all_attn_layers)],
                 'weight_decay': 0.0},
            ]
            optimizer_parameters = no_decay_optimizer_parameters + decay_optimizer_parameters + group_all_parameters

        else:
            optimizer_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

        return optimizer_parameters