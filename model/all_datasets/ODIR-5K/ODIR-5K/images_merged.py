
import pandas as pd
from dataset_index import ODIRImageIndex

base_dirs = [
    "ODIR-5K/Training Images",
    "ODIR-5K/Testing Images"
]

indexer = ODIRImageIndex(base_dirs)

df = pd.read_excel("data.xlsx")

df["filepath"] = df["image_name"].apply(indexer.get_path)

# remove missing images (VERY IMPORTANT for publication quality)
df = df.dropna(subset=["filepath"])
df["patient_id"] = df["image_name"].str.split("_").str[0]
label_cols = ["N","D","G","C","A","H","M","O"]

