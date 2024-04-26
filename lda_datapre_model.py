import pandas as pd
from itertools import product
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

df = pd.read_csv('daily_topics_心肌梗死.csv')
df = df[(df['Day'] != -1) & (df['Day'] != 16)]
# Group and merge order data
df_grouped = df.groupby(['Disease_Name', 'Day', 'Topic'])['Word'].apply(lambda x: set(x)).reset_index()

# Initialization MultiLabelBinarizer
mlb = MultiLabelBinarizer()
# Encoding a collection of 'Word' columns,Remove missing values
df_grouped = df_grouped.dropna(subset=['Word'])
df_grouped['Word'] = df_grouped['Word'].astype(str)
df_grouped['Word_Encoded'] = list(mlb.fit_transform(df_grouped['Word']))

df_grouped['Topic'] = df_grouped['Topic'].str.extract('(\d+)').astype(int)
df_grouped.sort_values(['Disease_Name', 'Day', 'Topic'], inplace=True)

# Calculate the top four topic combinations for each day for each disease
combinations_per_disease = {}
for disease, group in df_grouped.groupby('Disease_Name'):
    # Combined list of each disease, (day, topic)
    disease_combinations = []
    # Group each disease by day
    for day, day_group in group.groupby('Day'):
        top_topics = day_group['Topic'].unique()[:2]
        day_topics = [(day, topic) for topic in top_topics]
        disease_combinations.append(day_topics)
    # Arrange all possible tuple (day, topic) combination sequences of diseases
    combinations_per_disease[disease] = list(product(*disease_combinations))
# See all possible combinations of each disease
for disease, combinations in combinations_per_disease.items():
    print(f"Disease: {disease}, Total Combinations: {len(combinations)}")

# Create a new dictionary to store each path and corresponding vocabulary set for each disease
disease_paths_with_words = {}
for disease, combinations in combinations_per_disease.items():
    paths_with_words = []

    for combination in combinations:
        path_words = []

        for day_topic in combination:
            day, topic = day_topic
            words = df_grouped[(df_grouped['Disease_Name'] == disease) &
                               (df_grouped['Day'] == day) &
                               (df_grouped['Topic'] == topic)]['Word_Encoded'].values
            if len(words) > 0:
                path_words.append(words[0])
            else:
                path_words.append([])
        paths_with_words.append((combination, path_words))
    disease_paths_with_words[disease] = paths_with_words
print(disease_paths_with_words)

data_for_df = []
for disease, paths_with_words in disease_paths_with_words.items():
    for path, words in paths_with_words:
        words_list = [word.tolist() if isinstance(word, np.ndarray) else word for word in words]
        data_for_df.append([disease, path, words_list])

df = pd.DataFrame(data_for_df, columns=['Disease', 'Day_Topic_Combinations', 'Word_Encoded_List'])

# Save DataFrame to CSV file
df.to_csv('disease_paths_with_words_急性非ST段抬高.csv', index=False)
