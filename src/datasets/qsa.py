if __name__ == "__main__":
    import sys

    sys.path.append("../../")

import copy
import re
import json
from pathlib import Path
import requests
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


class QuestionStepsAnswerDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
    ):
        super().__init__()
        self.data = {}
        for idx, d in enumerate(data):
            self.data[idx] = {
                "idx": idx,
                "question": d["question"],
                "answer": d["answer"],
                "steps": "\n".join(d["steps"]),
                "n_steps": len(d["steps"]),
            }
        self.all_indices = list(self.data.keys())
        self.indices = copy.deepcopy(self.all_indices)

    def get_all_indices(self):
        return self.all_indices

    def set_indices(self, indices: List[int]):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        data_idx = self.indices[idx]
        return self.data[data_idx]


def _cot_to_steps(cot: str) -> List[str]:
    """Extract <<...>> reasoning steps from a CoT string."""
    steps = re.findall(r"<<[^>]+>>", cot)
    if steps:
        return steps
    # Fallback: split by whitespace when no <<>> markers found
    return [s.strip() for s in cot.split() if s.strip()]


class QSADataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, tiny_dataset=False, epoch_scaling=1, all_config=None, hf_dataset_id=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.use_hf = hf_dataset_id is not None
        if self.use_hf:
            self.hf_dataset_id = hf_dataset_id
        else:
            self.dataset_dir = Path(all_config.args.workspace_path, "datasets", "text_reasoning", dataset_name)
        self.tiny_dataset = tiny_dataset
        self.epoch_scaling = epoch_scaling
        self.all_config = all_config
        self.batch_size = all_config.dataloader.batch_size

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage: str = None):
        if self.use_hf:
            self._setup_from_hf(stage)
        else:
            self._setup_from_local(stage)

    # ========== Local JSON loading (original behaviour) ==========
    def _setup_from_local(self, stage: str = None):
        def load_split(split: str):
            with open(self.dataset_dir / f"{split}.json") as f:
                data = json.load(f)
                if self.tiny_dataset:
                    data = data[:32]
            return data

        if stage == "fit":
            self.train_set = self._create_dataset(load_split("train"), "train")
            self.val_set = self._create_dataset(load_split("val"), "val")
        elif stage == "test":
            self.test_set = self._create_dataset(load_split("test"), "test")

    # ========== HuggingFace dataset loading ==========
    def _setup_from_hf(self, stage: str = None):
        from datasets import load_dataset

        if stage == "fit":
            train_data = self._load_hf_train()
            if self.tiny_dataset:
                train_data = train_data[:32]
            # Split train into train (90%) / val (10%)
            n_val = max(1, int(len(train_data) * 0.1))
            val_data = train_data[-n_val:]
            train_data = train_data[:-n_val]
            self.train_set = self._create_dataset(train_data, "train")
            self.val_set = self._create_dataset(val_data, "val")
        elif stage == "test":
            test_data = self._load_hf_test()
            if self.tiny_dataset:
                test_data = test_data[:32]
            self.test_set = self._create_dataset(test_data, "test")

    def _load_hf_train(self) -> List[dict]:
        """Load training data from HuggingFace (list-of-dicts format)."""
        from datasets import load_dataset

        ds = load_dataset(self.hf_dataset_id, split="train", streaming=True)
        data = []
        for item in ds:
            data.append({
                "question": item["question"],
                "steps": _cot_to_steps(item["cot"]),
                "answer": item["answer"],
            })
        return data

    def _load_hf_test(self) -> List[dict]:
        """Load test data from HuggingFace.

        The test split uses a dict-of-lists (columnar) JSON format that the
        ``datasets`` library sometimes fails to parse, so we download and
        parse the raw JSON directly.
        """
        # Try the datasets library first, fall back to raw JSON
        try:
            from datasets import load_dataset

            ds = load_dataset(self.hf_dataset_id, split="test", streaming=True)
            data = []
            for item in ds:
                data.append({
                    "question": item["question"],
                    "steps": _cot_to_steps(item["cot"]),
                    "answer": item["answer"],
                })
            return data
        except Exception:
            pass

        # Fallback: download raw JSON from HF Hub
        url = f"https://huggingface.co/datasets/{self.hf_dataset_id}/resolve/main/gsm8k_test.json"
        resp = requests.get(url)
        resp.raise_for_status()
        raw = resp.json()  # dict of lists: {question: [...], cot: [...], answer: [...]}

        data = []
        n = len(raw["question"])
        for i in range(n):
            data.append({
                "question": raw["question"][i],
                "steps": _cot_to_steps(raw["cot"][i]),
                "answer": raw["answer"][i],
            })
        return data

    # ========== Shared helpers ==========
    def _create_dataset(self, raw_data: List[dict], mode: str) -> QuestionStepsAnswerDataset:
        return QuestionStepsAnswerDataset(
            data=raw_data,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.all_config.dataloader.batch_size,
            num_workers=self.all_config.dataloader.get("num_workers", 4),
            pin_memory=self.all_config.dataloader.get("pin_memory", True),
            persistent_workers=self.all_config.dataloader.get("persistent_workers", True),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.all_config.dataloader.get("val_batch_size", 1),
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.all_config.dataloader.get("val_batch_size", 1),
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

    def get_dataloader_to_filter_indices(self):
        return DataLoader(
            self.train_set,
            batch_size=8,
            shuffle=False,
        )

    def get_all_train_indices(self):
        return self.train_set.get_all_indices()

    def set_train_indices(self, train_indices):
        self.train_set.set_indices(train_indices)
