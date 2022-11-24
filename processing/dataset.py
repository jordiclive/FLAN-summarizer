import logging

import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    filename=f"HF_datasets_processing_logging",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.FATAL,
)


def filter(record):
    return record.levelno != logging.FATAL


logger = logging.getLogger(f"HF_datasets")


class HFDataset:
    def __init__(
        self,
        col_map,
        dataset_name,
        prompt="Summarize the following document:",
        dataset_id=None,
        dataset_keys=["train", "validation"],
        val_n=100,
        save=True,
    ):
        self.col_map = {"source": col_map[0], "target": col_map[1]}
        self.dataset_keys = dataset_keys
        self.dataset_name = dataset_name
        self.prompt = prompt
        self.val_n = val_n
        self.dataset_id = dataset_id
        self.save = save

    def rename_cols(self, df):
        col_map = dict(zip(self.col_map.values(), self.col_map.keys()))
        df = df.rename(col_map, axis=1)
        return df

    def save_format(self, train, val, parquet=True):
        if parquet:
            train.to_parquet(f"""train_{self.dataset_name.split('/')[0]}.parquet""")
            val.to_parquet(f"""val_{self.dataset_name.split('/')[0]}.parquet""")
        else:
            train.to_json(
                f"""train_{self.dataset_name.split('/')[0]}.json""", orient="split"
            )
            val.to_json(
                f"""val_{self.dataset_name.split('/')[0]}.json""", orient="split"
            )

    def apply_prompt(self, df):
        df["source"] = df["source"].apply(lambda x: self.prompt + x.strip())
        return df

    def process(self, train, val):
        return train, val

    def get_datasets(self, data_dir=None):
        if len(self.dataset_name.split("/")) == 2:
            name = self.dataset_name.split("/")
            df = datasets.load_dataset(name[0], name[1], data_dir=data_dir)
        else:
            df = datasets.load_dataset(self.dataset_name, data_dir=data_dir)
        try:
            if self.dataset_keys[1] not in df.keys():
                logger.fatal(f"{self.dataset_name} no validation")
                train = df[self.dataset_keys[0]].to_pandas()
                train, val = train_test_split(train, test_size=self.val_n)
            else:
                train = df[self.dataset_keys[0]].to_pandas()
                val = df[self.dataset_keys[1]].to_pandas().sample(self.val_n)
            train = self.rename_cols(train)
            val = self.rename_cols(val)
            train, val = self.process(train, val)
            train = self.apply_prompt(train)
            val = self.apply_prompt(val)
            train = train[["source", "target"]]
            val = val[["source", "target"]]
            if self.dataset_id is not None:
                train["dataset_id"] = self.dataset_id
                val["dataset_id"] = self.dataset_id
            if self.save:
                self.save_format(train, val)
                print(f"{self.dataset_name} length: {len(train)/1000}k")
                print(
                    f"""example: {train['source'][0]}
                        \n\n summary: {train['target'][0]}"""
                )
                logger.fatal(f"{self.dataset_name} length: {len(train)/1000}k")
                logger.fatal(
                    f"""example: {train['source'][0]}
                        \n\n summary: {train['target'][0]}"""
                )
            else:
                return train, val
        except:
            logger.fatal(f"PROBLEM with: {self.dataset_name}")
            return 0
