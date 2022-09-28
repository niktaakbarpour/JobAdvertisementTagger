import pandas as pd
from pandas import ExcelWriter


def read_cleaned_excel(file_name: str) -> pd.DataFrame:
    data = pd.read_excel(file_name)
    data["Keywords"] = data["Keywords"].apply(lambda keywords: set(keywords.split("|")))
    return data


def write_to_excel(data: pd.DataFrame, file_name: str):
    writer = ExcelWriter(file_name)
    data.to_excel(writer)
    writer.save()
