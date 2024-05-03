import json
import random
import os
from accelerate.logging import get_logger
from dataclasses import dataclass
from typing import Union, List
import torch

logger = get_logger(__name__, log_level="INFO")


datasets_list = ['new_modified_dataset_train',
'new_modified_dataset_dev',
'new_modified_dataset_test'
]


@dataclass
class DataSample:
    idx: int
    doc: str
    code: str
    # code_tokens: str
    # docstring_tokens: str
    # label: int
    retrieval_idx: int
    negative: str

class TrainSample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None,  label: Union[int, float] = 0):
        """
        Creates one TrainSample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<TrainSample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))


class Dataset(torch.utils.data.Dataset):
    def load_data(self, file_path: str = None):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class CoSQAData(Dataset):
    def __init__(
        self,
        dataset_name: str = "CosQA",
        split: str = "train",
        file_path: str = "cosqa",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        # logger.info(f"Loading CosQA data from {file_path}...")
        # file path is actually a directory

        data_map = {}
        all_samples = []
        id_ = 0
        for dataset in datasets_list:
            # logger.info(f"Loading dataset {dataset}...")
            if dataset not in data_map:
                data_map[dataset] = []
            with open(os.path.join(file_path, f"{dataset}.json"), "r") as f:
                dataset_samples = f.readlines()

            dataset_samples = [json.loads(d) for d in dataset_samples]

            for sample in dataset_samples:
                query = (
                    self.separator
                    + sample["code_tokens"]
                )
                if dataset in ['new_modified_dataset_train',
                    'new_modified_dataset_dev',
                    'new_modified_dataset_test'
                    ]:
                    pos = (
                        self.separator
                        + sample["docstring_tokens"]
                    )
                    neg = (
                        self.separator
                        + sample["negative"]
                    )
                else:
                    pos = self.separator + sample["docstring_tokens"]
                    neg = self.separator + sample["negative"]

                data_map[dataset].append(id_)

                all_samples.append(
                    DataSample(
                        retrieval_idx=id_,
                        code=query,
                        doc=pos,
                        negative=neg,
                        idx=dataset,
                    )
                )
                id_ += 1

        # combine split1 and split2
        new_data_map = {}
        for dataset in data_map:
            new_dataset = dataset.replace("_test", "").replace("_dev", "")
            if new_dataset not in new_data_map:
                new_data_map[new_dataset] = []
            new_data_map[new_dataset] += data_map[dataset]
        data_map = new_data_map

        if self.shuffle_individual_datasets:
            for task, samples in data_map.items():
                random.shuffle(samples)

        datasets = list(data_map.keys())

        logger.info(
            f"Batching Echo data properly for effective batch size of {self.effective_batch_size}..."
        )
        all_batches = []
        for dataset in datasets:
            dataset_samples = data_map[dataset]
            for i in range(0, len(dataset_samples), self.effective_batch_size):
                batch = dataset_samples[i : i + self.effective_batch_size]
                if len(batch) == self.effective_batch_size:
                    all_batches.append(batch)
                else:
                    print(f"Skip 1 batch for dataset {dataset}.")
                    logger.info(f"Skip 1 batch for dataset {dataset}.")
        random.shuffle(all_batches)

        final_idx_order = []
        for batch in all_batches:
            for idx in batch:
                final_idx_order.append(idx)

        self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples.")


    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.code, sample.doc, sample.negative], label=1.0
            )
        elif self.split == "validation":
            assert False, "Cosqa does not have a validation split."


def load_dataset(dataset_name, split="validation", file_path=None, **kwargs):
    """
    Loads a dataset by name.

    Args:
        dataset_name (str): Name of the dataset to load.
        split (str): Split of the dataset to load.
        file_path (str): Path to the dataset file.
    """
    dataset_mapping = {
        "CosQA":CoSQAData,
    }

    if dataset_name not in dataset_mapping:
        raise NotImplementedError(f"Dataset name {dataset_name} not supported.")

    if split not in ["train", "validation", "test"]:
        raise NotImplementedError(f"Split {split} not supported.")

    return dataset_mapping[dataset_name](
        split=split, file_path=file_path, **kwargs
    )