from datasets import Dataset, DatasetDict, concatenate_datasets
import json
import pandas as pd
import numpy as np

sem_label_dict = {
    'positive': 0,
    'negative': 1,
    'neutral': 2
}

label_count = {
    'laptop_sup': 3,
    'restaurant_sup': 3,
    'acl_sup': 6,
    'agnews_sup': 4,
    'laptop_fs': 3,
    'restaurant_fs': 3,
    'acl_fs': 6,
    'agnews_fs': 4
}

def prepare_few_shot_dataset(dataset_dict, num_samples):
    def sample_from_label(dataset, label, k):
        label_dataset = dataset.filter(lambda example: example['label'] == label)
        if len(label_dataset) > k:
            indices = np.random.choice(len(label_dataset), k, replace=False)
            return label_dataset.select(indices)
        return label_dataset

    for split in dataset_dict.keys():
        unique_labels = set(dataset_dict[split]['label'])
        sampled_datasets = [sample_from_label(dataset_dict[split], label, num_samples) for label in unique_labels]
        dataset_dict[split] = concatenate_datasets(sampled_datasets)

    return dataset_dict

def get_dataset(dataset_name, sep_token):
    '''
    dataset_name: str, the name of the dataset
    sep_token: str, the sep_token used by tokenizer (e.g., '<sep>')
    '''
    dataset = None

    if isinstance(dataset_name, list):
        total_datasets = {
            'train': [],
            'test': []
        }
        label_offset = 0
        
        for name in dataset_name:
            dataset = get_dataset(name, sep_token)
            for split in dataset.keys():
                adjusted_dataset = dataset[split].map(
                    lambda example: {'label': example['label'] + label_offset}, 
                    batched=False
                )
                total_datasets[split].append(adjusted_dataset)
            
            label_offset += label_count[name]
        
        for split in total_datasets.keys():
            total_datasets[split] = concatenate_datasets(total_datasets[split])
        print(total_datasets)
        return total_datasets

    if dataset_name == 'laptop_sup' or dataset_name == 'restaurant_sup' or dataset_name == 'laptop_fs' or dataset_name == 'restaurant_fs':
        if dataset_name == 'restaurant_sup':
            directory = './datasets/SemEval14-res'
        else:
            directory = './datasets/SemEval14-laptop'
        data_files = {
            'train': f'{directory}/train.json',
            'test': f'{directory}/test.json'
        }
        
        dataset_dict = {}
        for split, file_path in data_files.items():
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                texts, labels = [], []
                for entry in data.values():
                    text = entry['term'] + ' ' + entry['sentence']
                    label = sem_label_dict[entry['polarity']]
                    
                    texts.append(text)
                    labels.append(label)
                
                dataset_dict[split] = Dataset.from_dict({
                    'text': texts,
                    'label': labels
                })
    elif dataset_name == 'acl_sup' or dataset_name == 'acl_fs':
        directory = './datasets/acl-arc'
        data_files = {
            'train': f'{directory}/train.jsonl',
            'test': f'{directory}/test.jsonl'
        }
        
        dataset_dict = {}
        for split, file_path in data_files.items():
            with open(file_path, 'r', encoding='utf-8') as file:
                data = [json.loads(line) for line in file]
                
                texts, labels = [], []
                for entry in data:
                    text = entry['text']
                    label = entry['intent']
                    
                    texts.append(text)
                    labels.append(label)
                
                dataset_dict[split] = Dataset.from_dict({
                    'text': texts,
                    'label': labels
                })
    elif dataset_name == 'agnews_sup' or dataset_name == 'agnews_fs':
        directory = './datasets/archive'
        data_files = {
            'total': f'{directory}/test.csv'
        }
        
        dataset_dict = {}
        for split, file_path in data_files.items():
            with open(file_path, 'r', encoding='utf-8') as file:
                data = pd.read_csv(file)
                    
                texts, labels = [], []
                for _, entry in data.iterrows():
                    text = entry['Description']
                    label = int(entry['Class Index']) - 1

                    texts.append(text)
                    labels.append(label)
                
                dataset_dict[split] = Dataset.from_dict({
                    'text': texts,
                    'label': labels
                })
    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}')
    dataset = DatasetDict(dataset_dict)

    if dataset_name == 'agnews_sup' or dataset_name == 'agnews_fs':
        # split the test set into test and train
        dataset['train'] = dataset['total'].train_test_split(test_size=0.1, seed=2022)['train']
        dataset['test'] = dataset['total'].train_test_split(test_size=0.1, seed=2022)['test']
        dataset.pop('total')

    if dataset_name == 'laptop_fs' or dataset_name == 'restaurant_fs' or dataset_name == 'agnews_fs':
        # only take 32 samples/labels
        dataset = prepare_few_shot_dataset(dataset, 32)        

    if dataset_name == 'acl_fs':
        # only take 8 samples/labels
        dataset = prepare_few_shot_dataset(dataset, 8)

    return dataset

# print(get_dataset('laptop_sup', '<sep>'))
# print(get_dataset('restaurant_sup', '<sep>'))
# print(get_dataset('acl_sup', '<sep>'))
# print(get_dataset('agnews_sup', '<sep>'))
# print(get_dataset('laptop_fs', '<sep>'))
# print(get_dataset('restaurant_fs', '<sep>'))
# print(get_dataset('acl_fs', '<sep>'))
# print(get_dataset('agnews_fs', '<sep>'))

# print(get_dataset(['laptop_sup', 'restaurant_sup', 'acl_sup', 'agnews_sup'], '<sep>'))