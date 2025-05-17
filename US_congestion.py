import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os
import json

# 定义全局参数
CHUNKSIZE = 10000
RAW_PATH = r'D:\DataSet\us_congestion_2016_2022\us_congestion_2016_2022.csv'
OUTPUT_DIR = r'D:\DataSet\processed_csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 创建输出目录

# 预定义列名和编码器（确保跨分块一致性）
HIGH_MISSING_COLS = ['WindChill(F)', 'WindDir', 'WindSpeed(mph)']
NUM_COLS = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)']
CAT_COLS = ['City', 'Weather_Conditions']

# 预训练分类编码器（需全量数据统计）
# 若无法全量加载，需先遍历所有分块收集类别
all_weather = []
for chunk in pd.read_csv(RAW_PATH, chunksize=CHUNKSIZE, usecols=CAT_COLS):
    all_weather.extend(chunk['Weather_Conditions'].dropna().unique())
weather_encoder = LabelEncoder().fit(pd.Series(all_weather).astype(str))

# 分块处理主流程
chunk_iter = pd.read_csv(RAW_PATH, chunksize=CHUNKSIZE)

for i, chunk in enumerate(chunk_iter):
    print(f"Processing chunk {i + 1}...")

    # === 数据清洗 ===
    # 删除高缺失率列
    chunk_clean = chunk.drop(columns=HIGH_MISSING_COLS)

    # 填补数值列
    num_imputer = SimpleImputer(strategy='median')
    chunk_clean[NUM_COLS] = num_imputer.fit_transform(chunk_clean[NUM_COLS])

    # 填补分类列
    cat_imputer = SimpleImputer(strategy='most_frequent')
    chunk_clean[CAT_COLS] = cat_imputer.fit_transform(chunk_clean[CAT_COLS])

    # 删除时间缺失行
    chunk_clean = chunk_clean.dropna(subset=['StartTime', 'EndTime'])

    # === 时间处理 ===
    chunk_clean['StartTime'] = pd.to_datetime(chunk_clean['StartTime'], utc=True)
    chunk_clean['EndTime'] = pd.to_datetime(chunk_clean['EndTime'], utc=True)
    chunk_clean = chunk_clean.dropna(subset=['StartTime', 'EndTime'])  # 删除包含 NaT 值的行
    chunk_clean['Duration(mins)'] = (chunk_clean['EndTime'] - chunk_clean['StartTime']).dt.total_seconds() / 60

    # 过滤异常持续时间
    chunk_clean = chunk_clean[(chunk_clean['Duration(mins)'] > 0) &
                              (chunk_clean['Duration(mins)'] < 1440)]

    # === 特征工程 ===
    # 时间特征
    chunk_clean['StartHour'] = chunk_clean['StartTime'].dt.hour
    chunk_clean['StartDayOfWeek'] = chunk_clean['StartTime'].dt.weekday
    chunk_clean['IsWeekend'] = chunk_clean['StartDayOfWeek'].isin([5, 6]).astype(int)

    # 全局一致的分类编码
    chunk_clean['Weather_Encoded'] = weather_encoder.transform(
        chunk_clean['Weather_Conditions'].astype(str)
    )

    # === 保存分块 ===
    output_path = os.path.join(OUTPUT_DIR, f"processed_chunk_{i + 1}.csv")
    chunk_clean.to_csv(output_path, index=False)
    print(f"Saved chunk {i + 1} to {output_path}")

# 保存编码映射（示例：天气类别）
weather_mapping = dict(zip(weather_encoder.classes_, weather_encoder.transform(weather_encoder.classes_)))
with open("weather_mapping.json", "w") as f:
    json.dump(weather_mapping, f)

print("All chunks processed!")