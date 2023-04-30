import os
import argparse
import pandas as pd
import pickle
import json
from pathlib import Path

def map_label_to_index(input_data, l2id):

    n_labels_id = [l2id[nl] for nl in input_data['n_labels']]
    c_labels_id = [l2id[cl] for cl in input_data['c_labels']]

    input_data['n_labels'] = n_labels_id
    input_data['c_labels'] = c_labels_id

    return input_data

def convert2wrench(input_data):
    wrench_data_dict = dict()
    text = input_data['text']
    c_labels = input_data['c_labels']
    n_labels = input_data['n_labels']
    for idx, d in enumerate(input_data['text']):
        wrench_data_dict[str(idx)] = {'data': {'text': d}, 'label': c_labels[idx], 'weak_labels': [n_labels[idx]]}

    return wrench_data_dict


def save_wrench_json(wrench_format_data, output_dir, tag):
    file_path = Path(output_dir) / f"{tag}.json"
    with open(file_path, 'w') as fp:
        json.dump(wrench_format_data, fp)

def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/local/user_name_aabbcc/data/text_classification/Hausa')
    parser.add_argument('--save_root', type=str, default='/local/user_name_aabbcc/data/wrench/tmp_data/Hausa')
    parser.add_argument('--save_pickle', action='store_true')
    parser.add_argument('--save_txt', action='store_true')
    parser.add_argument('--save_wrench_json', action='store_true')
    args = parser.parse_args()

    Path(args.save_root).mkdir(parents=True, exist_ok=True)

    train_data = tsv2dict(dir=args.data_root, tag='train')
    val_data = tsv2dict(dir=args.data_root, tag='validation')
    test_data = tsv2dict(dir=args.data_root, tag='test')

    l2idx = {'World': 0, 'Nigeria': 1, 'Health': 2, 'Politics': 3, 'Africa': 4}
    idx2l = {v: k for k, v in l2idx.items()}

    l2idxHausa = {'Duniya': 0, 'Najeriya': 1, 'Lafiya': 2, 'Siyasa': 3, 'Afirka': 4}
    idx2lHausa = {v: k for k, v in l2idxHausa.items()}

    e2h = {'World': 'Duniya', 'Nigeria': 'Najeriya', 'Health': 'Lafiya', 'Politics': 'Siyasa', 'Africa': 'Afirka'}
    h2e = {v: k for k, v in e2h.items()}

    print('label mapping constructed')
    print(f'l2idx: {l2idx}')
    print(f'idx2l: {idx2l}')

    # The labels in the released dataset is in English, we use the English dict to convert labels
    train_data = convert2wrench(map_label_to_index(train_data, l2idx))
    val_data = convert2wrench(map_label_to_index(val_data, l2idx))
    test_data = convert2wrench(map_label_to_index(test_data, l2idx))

    # After Label conversion, we can/should save the original labels in Hausa.
    l2idx = l2idxHausa
    idx2l = idx2lHausa

    if args.save_wrench_json:
        save_wrench_json(train_data, args.save_root, tag='train')
        save_wrench_json(val_data, args.save_root, tag='valid')
        save_wrench_json(test_data, args.save_root, tag='test')
        save_wrench_json(idx2l, args.save_root, tag='label')

    # if args.save_pickle:
    #     save_dir = os.path.join(args.data_root, 'preprocessed')
    #     dict2pickle(train_c_dict, save_dir, 'train_clean.pickle')
    #     dict2pickle(train_cn_dict, save_dir, 'train_clean_noisy.pickle')
    #     dict2pickle(val_c_dict, save_dir, 'validation_clean.pickle')
    #     dict2pickle(val_cn_dict, save_dir, 'validation_clean_noisy.pickle')
    #     dict2pickle(test_c_dict, save_dir, 'test_clean.pickle')
    #     dict2pickle(test_cn_dict, save_dir, 'test_clean_noisy.pickle')
    #     dict2pickle(l2idx, save_dir, 'l2idx.pickle')
    #     dict2pickle(idx2l, save_dir, 'idx2l.pickle')
    #
    # if args.save_txt:
    #     save_dir = os.path.join(args.data_root, 'txt_data')
    #     list2txt(train_c_dict['text'], save_dir, 'train_clean.txt')
    #     list2txt(train_cn_dict['text'], save_dir, 'train_clean_noisy.txt')
    #     list2txt(val_c_dict['text'], save_dir, 'validation_clean.txt')
    #     list2txt(val_cn_dict['text'], save_dir, 'validation_clean_noisy.txt')
    #     list2txt(test_c_dict['text'], save_dir, 'test_clean.txt')
    #     list2txt(test_cn_dict['text'], save_dir, 'test_clean_noisy.txt')
    #
    #     list2pickle(train_c_dict['labels'], save_dir, 'train_clean_labels.pickle')
    #     list2pickle(train_cn_dict['labels'], save_dir, 'train_clean_noisy_labels.pickle')
    #     list2pickle(val_c_dict['labels'], save_dir, 'validation_clean_labels.pickle')
    #     list2pickle(val_cn_dict['labels'], save_dir, 'validation_clean_noisy_labels.pickle')
    #     list2pickle(test_c_dict['labels'], save_dir, 'test_clean_labels.pickle')
    #     list2pickle(test_cn_dict['labels'], save_dir, 'test_clean_noisy_labels.pickle')
    #
    #     dict2pickle(l2idx, save_dir, 'l2idx.pickle')
    #     dict2pickle(idx2l, save_dir, 'idx2l.pickle')

def list2txt(dict_content, save_dir, file_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'w') as filehandle:
        for listitem in dict_content:
            filehandle.write('%s\n' % listitem)

def list2pickle(input_list, save_dir, file_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as handle:
        pickle.dump(input_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_content_from_tsv(path):
    df = pd.read_csv(path, header=None, sep='\t')

    if len(df.columns) == 2:  # Hausa data
        content = df[0].values.tolist()[1:]
        labels = df[1].values.tolist()[1:]
    elif len(df.columns) == 3:  # Hausa noisy data
        content = df[1].values.tolist()[1:]
        labels = df[2].values.tolist()[1:]
    else:
        raise Exception(f"Unknown column number for Hausa dataset")

    return content, labels


def tsv2dict(dir, tag):
    clean_path = Path(dir) / f'{tag}_clean.tsv'
    noisy_path = Path(dir) / f'{tag}_clean_noisy.tsv'
    content1, c_labels = get_content_from_tsv(clean_path)
    content2, n_labels = get_content_from_tsv(noisy_path)

    # sanity checks
    assert len(content1) == len(content2)
    for c1, c2 in zip(content1, content2):
        assert c1 == c2

    assert len(c_labels) == len(n_labels) == len(content1) == len(content2)

    return {'text': content1, 'c_labels': c_labels, 'n_labels': n_labels}



def dict2tsv(dict_content, file_name, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f'{file_name}.tsv')

    text = dict_content["text"]

    labels = dict_content["labels"]

    with open(save_path, 'w') as writer:
        for t, l in zip(text, labels):
            writer.write(f'{t}\t{l}\n')


if __name__=='__main__':
    preprocess()