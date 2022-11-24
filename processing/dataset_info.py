import glob

import pandas as pd

from dataset import HFDataset


class Scitldr(HFDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, train, val):
        for df in [train, val]:
            df["source"] = df["source"].apply(lambda x: "\n".join(x))
            df["target"] = df["target"].apply(lambda x: "\n".join(x))
        return train, val


def process_datasets(df, val_n=100):
    for i, row in df.iterrows():
        dataset_name = row["hf_dataset_key"]
        hf_dataset = HFDataset(
            col_map=[row["source_key"], row["target_key"]],
            dataset_name=row["hf_dataset_key"],
            prompt=row["flan_prompt"],
            val_n=val_n,
        )
        if dataset_name == "wikihow/all":
            # Need to manually download wikihowAll.csv and place in directory
            hf_dataset.get_datasets(data_dir=".")
        elif "scitldr" in dataset_name:
            # specific processing
            hf_dataset = Scitldr(
                col_map=[row["source_key"], row["target_key"]],
                dataset_name=row["hf_dataset_key"],
                prompt=row["flan_prompt"],
                val_n=val_n,
            )
            hf_dataset.get_datasets()
        else:
            hf_dataset.get_datasets()


def merge(dataset_items):
    for i, dataset_item in enumerate(dataset_items):
        if i == 0:
            df = pd.read_parquet(dataset_item)

        else:
            new_item = pd.read_parquet(dataset_item)
            df = pd.concat([df, new_item])

    df.reset_index(inplace=True, drop=True)
    return df


def merge_datasets():
    train_sets = glob.glob("train*")
    train = merge(train_sets)
    train.to_parquet("train.parquet")

    val_sets = glob.glob("val*")
    val = merge(val_sets)
    val.to_parquet("val.parquet")


if __name__ == "__main__":
    # df = pd.read_csv("hf_datasets.csv")
    # process_datasets(df)
    merge_datasets()


    # xsum = {
    #     "hf_dataset_key": "xsum",
    #     "source_target_keys": ["document", "summary"],
    #     "flan_prompt": "Given the following news article, summarize the article in one sentence: ",
    # }
    #
    # cnn = {
    #     "hf_dataset_key": "cnn_dailymail/3.0.0",
    #     "source_target_keys": ["article", "highlights"],
    #     "flan_prompt": "Produce an article summary of the following news article: ",
    # }
    #
    # samsum = {
    #     "hf_dataset_key": "samsum",
    #     "source_target_keys": ["dialogue", "summary"],
    #     "flan_prompt": "Briefly summarize in third person the following conversation: ",
    # }
    # billsum = {
    #     "hf_dataset_key": "billsum",
    #     "source_target_keys": ["text", "summary"],
    #     "flan_prompt": "Summarize the following proposed legislation (bill): ",
    # }
    # wikihow = {
    #     "hf_dataset_key": "wikihow/all",
    #     "source_target_keys": ["text", "headline"],
    #     "flan_prompt": "Produce an article summary including outlines of each paragraph of the following article: ",
    # }
    # scitldr = {
    #     "hf_dataset_key": "scitldr/AIC",
    #     "source_target_keys": ["source", "target"],
    #     "flan_prompt": "Given the following scientific article, provide a TL;DR summary: ",
    # }
    #
    #
    # df = pd.DataFrame({"hf_dataset_key":[],"source_target_keys":[],"flan_prompt":[]})
    # df = [wikihow,xsum,cnn,samsum,scitldr,billsum]
    # df = pd.DataFrame(df)
    # df['source_key'] = df['source_target_keys'].apply(lambda x: x[0])
    # df['target_key'] = df['source_target_keys'].apply(lambda x: x[1])
    # df = df[["hf_dataset_key","source_key","target_key","flan_prompt"]]
    # df.to_csv('hf_datasets.csv',index=False)

    # dataset_name =  'billsum'
    # PROMPT = 'Summarize the following proposed legislation (bill): '
    # COL_MAP = ['text', "summary"]
    # DATASET_KEYS = ['train','test']
    # x = HFDataset(col_map=COL_MAP, dataset_name=dataset_name, prompt=PROMPT,dataset_keys=DATASET_KEYS)
    # x.get_datasets()
    # }
    #
    # dataset_name = 'scitldr/AIC'
    # PROMPT = "Given the following scientific article, provide a TL;DR summary: "
    # COL_MAP = ['source', "target"]
    # x = scitldr(col_map=COL_MAP, dataset_name=dataset_name, prompt=PROMPT)
    # x.get_datasets()
    # hf_keys = ['samsum','xsum','scitldr']

    # dataset_name = 'samsum'
    # PROMPT = "Briefly summarize in third person the following conversation: "
    # COL_MAP = ['dialogue', "summary"]
    #
    # PROMPT = "Briefly summarize in third person the following conversation: "
    # COL_MAP = ['dialogue',"summary"]}
    #
    # dataset_name = 'cnn_dailymail/3.0.0'
    # PROMPT = 'Produce an article summary of the following news article: '
    # COL_MAP = ['article',"highlights"]

    # DATASET_NAME = 'wikihow'
    # PROMPT = "Produce an article summary including outlines of each paragraph of the following article: "
    # COL_MAP = ['text','headline']
    # dataset_name = "wikihow/all"
    # wikihow = HFDataset(col_map=COL_MAP,dataset_name= dataset_name,prompt=PROMPT)
    # wikihow.get_datasets(data_dir='.')
    #
    # dataset_name = 'xsum'
    # PROMPT = 'Given the following news article, summarize the article in one sentence: '
    # COL_MAP = ['document', "summary"]
    # x = HFDataset(col_map=COL_MAP,dataset_name= dataset_name,prompt=PROMPT)
    # x.get_datasets()
    #
    # dataset_name = "cnn_dailymail/3.0.0"
    # PROMPT = "Produce an article summary of the following news article: "
    # COL_MAP = ["article", "highlights"]
    #
    # x = HFDataset(col_map=COL_MAP, dataset_name=dataset_name, prompt=PROMPT)
    # x.get_datasets()
    #
    # dataset_name = "samsum"
    # PROMPT = "Briefly summarize in third person the following conversation: "
    # COL_MAP = ["dialogue", "summary"]
    # x = HFDataset(col_map=COL_MAP, dataset_name=dataset_name, prompt=PROMPT)
    # x.get_datasets()
    #
    # dataset_name = "scitldr/AIC"
    # PROMPT = "Given the following scientific article, provide a TL;DR summary: "
    # COL_MAP = ["source", "target"]
    # x = scitldr(col_map=COL_MAP, dataset_name=dataset_name, prompt=PROMPT)
    # x.get_datasets()
    # hf_keys = ["samsum", "xsum", "scitldr"]
    #
    # dataset_name = "billsum"
    # PROMPT = "Summarize the following proposed legislation (bill): "
    # COL_MAP = ["text", "summary"]
    # DATASET_KEYS = ["train", "test"]
    # x = HFDataset(
    #     col_map=COL_MAP,
    #     dataset_name=dataset_name,
    #     prompt=PROMPT,
    #     dataset_keys=DATASET_KEYS,
    # )
    # x.get_datasets()

    # train = pd.read_parquet('df1.parquet', engine='pyarrow' )
    # df2 = pd.read_parquet('df2.parquet',engine='pyarrow')
    # train.columns = ['source','target']
    # df2.columns = ['source','target']
    # train = pd.concat([train,df2])
    # train['source'] = train['source'].apply(lambda x: 'Give a TL;DR Summary of the following: ' + x.strip())
    # train.reset_index(inplace=True,drop=True)
    # train, val = train_test_split(train, test_size=100)
