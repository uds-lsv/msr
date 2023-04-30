import numpy as np
from snorkel.utils import probs_to_preds
from sklearn.metrics import classification_report

class MajorityVoting:
    def __init__(self, **kwargs):
        super().__init__()

    def predict_proba_ner(self, dataset, weight=None, ABSTAIN=-1) -> np.ndarray:
        proba_list = []
        weak_labels = dataset.weak_labels
        n_class = dataset.num_classes

        for weak_label in weak_labels:
            L = np.array(weak_label)
            weight = np.ones_like(L)
            n, m = L.shape
            Y_p = np.zeros((n, n_class))
            for i in range(n):
                counts = np.zeros(n_class)
                for j in range(m):
                    if L[i, j] != ABSTAIN:
                        counts[L[i, j]] += 1 * weight[i, j]
                # Y_p[i, :] = np.where(counts == max(counts), 1, 0)
                if counts.sum() == 0:
                    counts += 1
                Y_p[i, :] = counts
            Y_p /= Y_p.sum(axis=1, keepdims=True)
            proba_list.append(Y_p)

        return proba_list




    def predict_proba(self, dataset, weight=None, ABSTAIN=-1) -> np.ndarray:
        # this function gets the probability of the noisy labels
        # need to call the Snokerl's
        L = np.array(dataset.weak_labels)
        if weight is None:
            weight = np.ones_like(L)

        n_class = dataset.num_classes
        n, m = L.shape
        Y_p = np.zeros((n, n_class))
        for i in range(n):
            counts = np.zeros(n_class)
            for j in range(m):
                if L[i, j] != ABSTAIN:
                    counts[L[i, j]] += 1 * weight[i, j]
            # Y_p[i, :] = np.where(counts == max(counts), 1, 0)
            if counts.sum() == 0:
                counts += 1
            Y_p[i, :] = counts
        Y_p /= Y_p.sum(axis=1, keepdims=True)

        return Y_p

    def flatten(self, probs):
        L = []  # weak labels
        indexes = [0]
        for i in range(len(probs)):
            L += list(probs[i])
            indexes.append(len(probs[i]))
        indexes = np.cumsum(indexes)
        return np.array(L), indexes

    def predict(self, dataset, **kwargs) -> np.ndarray:
        """Method for predicting on given dataset.

        Parameters
        ----------
        """

        if kwargs['task_type']=='ner': # ner task
            probas = self.predict_proba_ner(dataset)
            # actually, flatten is not that necessary
            # but please do it, since the tie function in probs_to_preds needs the correct length as random seed
            # flatten used for reproducibility
            probas_flatten, indexes = self.flatten(probas)
            majority_preds_flatten = probs_to_preds(probs=np.array(probas_flatten))
            majority_preds = [list(majority_preds_flatten[start:end]) for (start, end) in zip(indexes[:-1], indexes[1:])]
            return majority_preds
        elif kwargs['task_type']=='text_cls':
            proba = self.predict_proba(dataset)

            return probs_to_preds(probs=proba)
        else:
            raise ValueError("unknown task type")

    def test(self, args, dataset, y_true=None, strict=True, **kwargs):
        if y_true is None:
            y_true = dataset.labels
        preds = self.predict(dataset, **kwargs)
        classification_score = classification_report(y_true, preds,
                                                     target_names=dataset.label_txt_list, output_dict=True)

        return classification_score
