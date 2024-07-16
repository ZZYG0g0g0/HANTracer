import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_true_links(file_path):
    true_links = []
    with open(file_path, 'r') as file:
        for line in file:
            req, code = line.strip().split()
            true_links.append((req, code))
    return true_links

def compute_f1(true_links, sorted_similarity_df, threshold_index):
    true_num = 0
    predicted_labels = np.zeros(len(sorted_similarity_df), dtype=int)
    for index in range(threshold_index):
        req = sorted_similarity_df.iloc[index]['Requirement']
        code = sorted_similarity_df.iloc[index]['Code']
        if (req, code) in true_links:
            true_num += 1
            predicted_labels[index] = 1
    precision = true_num / threshold_index
    recall = true_num / len(true_links)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return f1

if __name__ == '__main__':
    embedding_methods = ['albert', 'bert', 'roberta', 'word2vec', 'xlnet']
    datasets = ['Derby', 'Drools', 'Infinispan', 'iTrust', 'Maven', 'Pig', 'Seam2']
    emb_column = []
    dataset_column = []
    f1_column = []
    threshold_column = []

    for embedding_method in embedding_methods:
        for dataset in datasets:
            # 读取嵌入向量
            requirement_embeddings = pd.read_excel(f'../docs/{dataset}/uc/uc_{embedding_method}_vectors.xlsx', header=None, skiprows=1).values
            code_embeddings = pd.read_excel(f'../docs/{dataset}/cc/cc_{embedding_method}_vectors.xlsx', header=None, skiprows=1).values

            # 计算余弦相似度
            similarity_matrix = cosine_similarity(requirement_embeddings, code_embeddings)

            # 展开相似度矩阵
            similarity_scores = similarity_matrix.flatten()

            # 加载真实标签
            true_links = load_true_links(f'../dataset/{dataset}/true_set.txt')

            # 获取需求和代码的数量
            num_requirements, num_codes = requirement_embeddings.shape[0], code_embeddings.shape[0]
            req_names = sorted(os.listdir(f'../dataset/{dataset}/uc'))
            code_names = sorted(os.listdir(f'../dataset/{dataset}/cc'))
            req_name_column = [req_names[i // num_codes] for i in range(num_requirements * num_codes)]
            code_name_column = [code_names[i % num_codes] for i in range(num_requirements * num_codes)]

            similarity_df = pd.DataFrame({
                'Requirement': req_name_column,
                'Code': code_name_column,
                'Similarity': similarity_scores
            })

            # 排序相似度分数
            sorted_similarity_df = similarity_df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)

            # 评估每个百分比阈值下的精确率、召回率和F1值
            best_f1 = 0
            best_threshold = 0
            percentiles = np.arange(0.01, 1.01, 0.01)

            for percentile in percentiles:
                print(f'{embedding_method}->{dataset}->{percentile}')
                threshold_index = int(percentile * len(sorted_similarity_df))
                f1 = compute_f1(true_links, sorted_similarity_df, threshold_index)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = percentile

            threshold_column.append(best_threshold)
            f1_column.append(best_f1)
            emb_column.append(embedding_method)
            dataset_column.append(dataset)

    res_df = pd.DataFrame({
        'embedding': emb_column,
        'dataset': dataset_column,
        'f1': f1_column,
        'threshold': threshold_column
    })
    res_df.to_excel('RQ1_embedding.xlsx', index=False)
