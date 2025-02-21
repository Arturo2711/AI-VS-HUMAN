import pandas as pd 
import numpy as np


def generate_balanced_data_csv(df, num_instances_per_class):
    class_human = df[df['label'] == 0]
    class_machine = df[df['label'] == 1]
    new_class_human = class_human.sample(n=num_instances_per_class, replace=False, random_state=42, axis=0)
    new_class_machine = class_machine.sample(n=num_instances_per_class, replace=False, random_state=42, axis=0)
    balanced_df = pd.concat([new_class_human, new_class_machine])
    return balanced_df


df = pd.read_csv('Data and exploratory analysis/data_cleaned.csv')
balanced_df = generate_balanced_data_csv(df, num_instances_per_class=163059)
print(balanced_df['label'].value_counts())

balanced_df.to_csv('Data and exploratory analysis/balanced_data_big.csv', index=False)