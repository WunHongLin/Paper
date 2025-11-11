import kagglehub
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import chain

def GetAAPD():
    """
    從KAGGLE上抓取對應的AAPD資料集並存取其路徑
    """
    path = kagglehub.dataset_download("xiaojuanwang9/aapd-dataset")

    """
    用一個迴圈去抓取訓練集跟測試集的資料，並個別將其存放在對應的aapd資料集中，以便後續進行使用
    """
    for TrainingSet in ["test", "train", "val"]:
        SetPath = f"{path}/{TrainingSet}.txt"
        Texts, Labels = [], []
        with open(SetPath, "r", encoding="utf-8") as f:
            for index, line in enumerate(f.readlines()):
                if index % 2 == 0:
                    Texts.append(line)
                else:
                    Labels.append(line)

        """
        確認當前檔案是否有資料夾，若沒有建立一個對應檔案
        """
        if not os.path.exists(f"./AAPD/{TrainingSet}"):
            os.makedirs(f"./AAPD/{TrainingSet}/")
            SetDF = pd.DataFrame({"Texts": Texts, "Labels": Labels})
            SetDF.to_csv(f"./AAPD/{TrainingSet}/{TrainingSet}.csv", index=False)

        print(f"{TrainingSet}資料集已完成儲存")

        """
        將剩餘檔案轉移到同等資料夾中
        """
        shutil.copy(f"{path}\\label_to_index.json", f"./AAPD/label_to_index.json")
        shutil.copy(f"{path}\\vocab.txt", f"./AAPD/vocab.txt")

def GetEUR():
    """
    從KAGGLE上抓取對應的EUR-Lex資料集並存取其路徑
    """
    path = kagglehub.dataset_download("puskas78/eurlex-dataset")

    """
    抓取EUR-all 這一份檔案，並存取其act_raw_text 以及 Subject_matter 這兩個欄位
    """
    SetPath = f"{path}/EurLex_all.csv"
    SetDF = pd.read_csv(SetPath)[["act_raw_text", "Subject_matter"]].dropna()
    SetDF.columns = ['Texts', 'Labels']

    """
    將抓取到的資料進行切割，分成訓練集、驗證集、測試集
    """
    train_df, test_df = train_test_split(SetDF, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    """
    確認當前檔案是否有資料夾，若沒有建立一個對應檔案
    """
    if not os.path.exists(f"./EUR-Lex/train"):
        os.makedirs(f"./EUR-Lex/train/")
        train_df.to_csv(f"./EUR-Lex/train/train.csv", index=False)
    
    if not os.path.exists(f"./EUR-Lex/val"):
        os.makedirs(f"./EUR-Lex/val/")
        val_df.to_csv(f"./EUR-Lex/val/val.csv", index=False)

    if not os.path.exists(f"./EUR-Lex/test"):
        os.makedirs(f"./EUR-Lex/test/")
        test_df.to_csv(f"./EUR-Lex/test/test.csv", index=False)

    print("EUR-Lex資料集已完成儲存")

"""
將AAPD的標籤轉成數字並儲存到新檔案中
"""
def PreprocessAAPDLabels():
    """
    首先先將.JSON檔案讀取進來
    """
    LabelToIndex = pd.read_json("./AAPD/label_to_index.json", typ='series').to_dict()
    """
    將train、val、test整合後，按照8:1:1 比例分開來
    """
    train_df = pd.read_csv("AAPD/train/train.csv")
    val_df = pd.read_csv("AAPD/val/val.csv")
    test_df = pd.read_csv("AAPD/test/test.csv")
    total_df = pd.concat([train_df, val_df, test_df])

    train_df, test_df = train_test_split(total_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    train_df.to_csv("AAPD/train/train.csv", index=False)
    val_df.to_csv("AAPD/val/val.csv", index=False)
    test_df.to_csv("AAPD/test/test.csv", index=False)

    """
    應用迴圈去取每一份資料及的label欄位
    """
    for TrainingSet in ["train", "val", "test"]:
        SetDF = pd.read_csv(f"./AAPD/{TrainingSet}/{TrainingSet}.csv")
        LabelDF = SetDF["Labels"]
        """
        處理每一筆資料的標籤，將其轉成數字並存放在一個list中
        """
        NewLabelList = []
        NewDF = SetDF.copy()
        for Labels in LabelDF:
            Labels = Labels.replace("\n", "").split(" ")
            NewLabels = " ".join([str(LabelToIndex[label]) for label in Labels])
            NewLabelList.append(NewLabels)

        NewDF["NumberLabels"] = NewLabelList
        NewDF.to_csv(f"./AAPD/{TrainingSet}/{TrainingSet}.csv", index=False)   

"""
將EUR-Lex的標籤轉成數字並儲存到新檔案中
"""
def PreprocessEURLabels():
    """
    因為EUR並不包含標籤對應數字的JSON檔案，因此這裡手動進行
    """
    df = pd.read_csv("./EUR-Lex/train/train.csv")
    UniqueLabels = set(chain.from_iterable(df["Labels"].str.split("; ")))
    Label2ID = {label: i for i, label in enumerate(sorted(UniqueLabels))}
    """
    處理每一筆資料的標籤，將其轉成數字並存放在一個list中
    """
    for TrainingSet in ["train", "val", "test"]:
        SetDF = pd.read_csv(f"./EUR-Lex/{TrainingSet}/{TrainingSet}.csv")
        LabelDF = SetDF["Labels"]
        """
        處理每一筆資料的標籤，將其轉成數字並存放在一個list中
        """
        NewLabelList = []
        NewDF = SetDF.copy()
        for Labels in LabelDF:
            Labels = Labels.split("; ")
            NewLabels = " ".join([str(Label2ID[label]) for label in Labels])
            NewLabelList.append(NewLabels)

        NewDF["NumberLabels"] = NewLabelList
        NewDF.to_csv(f"./EUR-Lex/{TrainingSet}/{TrainingSet}.csv", index=False)   

"""
根據呼叫數值決定要做甚麼動作
"""
def DataPreprecessed(Dataset):
    if Dataset == "AAPD":
        GetAAPD()
        """
        這裡需要建立另外一個函式去將文字標籤轉成數字並將其加在另外一個欄位
        """
        PreprocessAAPDLabels()
    elif Dataset == "EUR-Lex":
        GetEUR()
        """
        這裡需要建立另外一個函式去將文字標籤轉成數字並將其加在另外一個欄位
        """
        PreprocessEURLabels()

DataPreprecessed("AAPD")