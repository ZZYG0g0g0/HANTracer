import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HANConv
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class HAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, metadata):
        super(HAN, self).__init__()
        self.conv = HANConv(in_channels, hidden_channels, metadata, heads=8)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)

        for key in x_dict.keys():
            x = x_dict[key]

            # 批量归一化和激活
            x = self.bn1(x)
            x_dict[key] = x

        return x_dict


class Classifier(torch.nn.Module):
    def forward(self, x_req: Tensor, x_code: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_req = x_req[edge_label_index[0]]
        edge_feat_code = x_code[edge_label_index[1]]
        return torch.sigmoid((edge_feat_req * edge_feat_code).sum(dim=-1))


class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata):
        super(Model, self).__init__()
        self.han = HAN(in_channels, out_channels, metadata)
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "req": data["req"].x,
            "code": data["code"].x,
        }
        x_dict = self.han(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["req"],
            x_dict["code"],
            data["req", "link", "code"].edge_label_index,
        )
        return pred


def generate_req_code_edge(dataset_name):
    uc_names = os.listdir('../dataset/' + dataset_name + '/uc')
    cc_names = os.listdir('../dataset/' + dataset_name + '/cc')
    uc_idx_dict = {uc_names[i]: i for i in range(len(uc_names))}
    cc_idx_dict = {cc_names[j].split('.')[0]: j for j in range(len(cc_names))}
    edge_from, edge_to = [], []
    with open('../dataset/' + dataset_name + "/true_set.txt", 'r', encoding='ISO8859-1') as df:
        lines = df.readlines()
    for line in lines:
        uc_name, cc_name = line.split(' ')[0], line.split(' ')[1].split('.')[0]
        if uc_name in uc_idx_dict and cc_name in cc_idx_dict:
            edge_from.append(uc_idx_dict[uc_name])
            edge_to.append(cc_idx_dict[cc_name])
        else:
            print(f"Error: {uc_name} or {cc_name} not found in dictionaries.")
    return edge_from, edge_to

if __name__ == '__main__':
    datasets = ['Derby', 'Drools', 'Infinispan', 'iTrust', 'maven', 'Pig', 'Seam2']
    nodes_features = ['word2vec', 'bert', 'albert', 'roberta', 'xlnet']
    for nodes_feature in nodes_features:
        all_results = []
        for dataset in datasets:
            # Repeat the experiment 50 times
            precision_scores = []
            recall_scores = []
            f1_scores = []
            uc_df = pd.read_excel(f'../docs/{dataset}/uc/uc_{nodes_feature}_vectors.xlsx')
            cc_df = pd.read_excel(f'../docs/{dataset}/cc/cc_{nodes_feature}_vectors.xlsx')
            req_feat = torch.from_numpy(uc_df.values).to(torch.float)
            code_feat = torch.from_numpy(cc_df.values).to(torch.float)

            edge_from, edge_to = generate_req_code_edge(dataset)
            edge_index = torch.tensor([edge_from, edge_to], dtype=torch.long)

            data = HeteroData()
            data["req"].x = req_feat
            data["code"].x = code_feat
            data["req", "link", "code"].edge_index = edge_index
            data = T.ToUndirected()(data)

            for i in range(50):
                print(f"Experiment {i+1}")
                transform = T.RandomLinkSplit(
                    num_val=0.1,
                    num_test=0.1,
                    disjoint_train_ratio=0.3,
                    neg_sampling_ratio=2.0,
                    add_negative_train_samples=False,
                    edge_types=("req", "link", "code"),
                    rev_edge_types=("code", "rev_link", "req")
                )
                train_data, val_data, test_data = transform(data)

                train_loader = LinkNeighborLoader(
                    data=train_data,
                    num_neighbors=[20, 10],
                    neg_sampling_ratio=2.0,
                    edge_label_index=(("req", "link", "code"), train_data["req", "link", "code"].edge_label_index),
                    edge_label=train_data["req", "link", "code"].edge_label,
                    batch_size=128,
                    shuffle=True,
                )
                model = Model(in_channels=req_feat.size(1),  out_channels=128, metadata=data.metadata())
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(device)
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(1, 30):
                    total_loss = total_examples = 0
                    for sampled_data in tqdm.tqdm(train_loader):
                        try:
                            optimizer.zero_grad()
                            sampled_data.to(device)
                            pred = model(sampled_data)
                            ground_truth = sampled_data["req", "link", "code"].edge_label
                            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                            loss.backward()
                            optimizer.step()
                            total_loss += float(loss) * pred.numel()
                            total_examples += pred.numel()
                        except IndexError as e:
                            print(f"IndexError: {e}")
                            print(f"Sampled Data: {sampled_data}")
                            break
                    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

                val_loader = LinkNeighborLoader(
                    data=val_data,
                    num_neighbors=[20, 10],
                    edge_label_index=(("req", "link", "code"), val_data["req", "link", "code"].edge_label_index),
                    edge_label=val_data["req", "link", "code"].edge_label,
                    batch_size=3 * 128,
                    shuffle=False,
                )

                preds = []
                ground_truths = []
                for sampled_data in tqdm.tqdm(val_loader):
                    with torch.no_grad():
                        sampled_data.to(device)
                        preds.append(model(sampled_data))
                        ground_truths.append(sampled_data["req", "link", "code"].edge_label)
                pred = torch.cat(preds, dim=0).cpu().numpy()
                ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
                pred_labels = (pred > 0.5).astype(np.float32)

                precision = precision_score(ground_truth, pred_labels, average='binary')
                recall = recall_score(ground_truth, pred_labels, average='binary')
                f1 = f1_score(ground_truth, pred_labels, average='binary')

                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

                print(f"precision: {precision:.4f}\nrecall: {recall:.4f}\nf1: {f1:.4f}")

            # 计算平均值
            avg_precision = np.mean(precision_scores)
            avg_recall = np.mean(recall_scores)
            avg_f1 = np.mean(f1_scores)

            print(f"Average Precision: {avg_precision:.4f}")
            print(f"Average Recall: {avg_recall:.4f}")
            print(f"Average F1: {avg_f1:.4f}")

            all_results.append({
                'Dataset': dataset,
                'Model': nodes_feature,
                'Precision': avg_precision,
                'Recall': avg_recall,
                'F1': avg_f1
            })
        # 将所有结果写入Excel
        results_df = pd.DataFrame(all_results)
        results_df.to_excel(f'RQ1_{nodes_feature}.xlsx', index=False)