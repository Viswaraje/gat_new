import torch
import time
import argparse
import logging
import torch.nn as nn
from model import *
from data_process import process
from my_utiils import *
from data_load import dataload

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument Parsing
parser = argparse.ArgumentParser(description='Drug Response Prediction')
parser.add_argument('--alph', type=float, default=0.30, help='Alpha weight for loss function')
parser.add_argument('--beta', type=float, default=0.30, help='Beta weight for loss function')
parser.add_argument('--epoch', type=int, default=350, help='Number of training epochs')
parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden layer dimensions')
parser.add_argument('--output_channels', type=int, default=100, help='Output feature dimensions')
parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
args = parser.parse_args()

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}')

start_time = time.time()

# Data Files
Drug_info_file = '../data/Drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
IC50_threds_file = '../data/Drug/drug_threshold.txt'
Drug_feature_file = '../data/Drug/drug_graph_feat'
Cell_line_info_file = '../data/Celline/Cell_lines_annotations.txt'
Genomic_mutation_file = '../data/Celline/genomic_mutation_34673_demap_features.csv'
Cancer_response_exp_file = '../data/Celline/GDSC_IC50.csv'
Gene_expression_file = '../data/Celline/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = '../data/Celline/genomic_methylation_561celllines_808genes_demap_features.csv'

# Load Data
drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs = \
    dataload(Drug_info_file, IC50_threds_file, Drug_feature_file, Cell_line_info_file,
             Genomic_mutation_file, Cancer_response_exp_file, Gene_expression_file, Methylation_file)

# Process Data
drug_set, cellline_set, train_edge, label_pos, train_mask, test_mask, atom_shape = \
    process(drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs)

# Convert to Tensor and Move to Device
label_pos, train_mask, test_mask = map(torch.tensor, (label_pos, train_mask, test_mask))
label_pos, train_mask, test_mask = label_pos.to(device), train_mask.to(device), test_mask.to(device)

# Model Initialization
encoder_output_dim = args.hidden_channels * args.heads  # 256 * 4 = 1024
model = GraphCDR(
    hidden_channels=args.hidden_channels,
    encoder=Encoder(args.output_channels, args.hidden_channels, heads=args.heads),
    summary=Summary(args.output_channels, encoder_output_dim),
    feat=NodeRepresentation(atom_shape, gexpr_feature.shape[-1], methylation_feature.shape[-1], args.output_channels),
    index=nb_celllines
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
myloss = nn.BCELoss()

# Loss Computation Function
def compute_loss(drug, cell):
    pos_z, neg_z, summary_pos, summary_neg, pos_adj = model(drug.x.to(device), drug.edge_index.to(device), drug.batch.to(device),
                                                             torch.tensor(cell[0]).to(device),
                                                             torch.tensor(cell[1]).to(device),
                                                             torch.tensor(cell[2]).to(device),
                                                             torch.tensor(train_edge).to(device))
    dgi_pos = model.loss(pos_z, neg_z, summary_pos)
    dgi_neg = model.loss(neg_z, pos_z, summary_neg)
    pos_loss = myloss(pos_adj[train_mask], label_pos[train_mask])
    loss = (1 - args.alph - args.beta) * pos_loss + args.alph * dgi_pos + args.beta * dgi_neg
    return loss, pos_adj

# Training Function
def train():
    model.train()
    total_loss = 0
    for drug, cell in zip(drug_set, cellline_set):
        optimizer.zero_grad()
        loss, _ = compute_loss(drug, cell)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    logging.info(f'Train Loss: {round(total_loss, 4)}')

# Testing Function
def test():
    model.eval()
    with torch.no_grad():
        for drug, cell in zip(drug_set, cellline_set):
            _, pre_adj = compute_loss(drug, cell)
            loss = myloss(pre_adj[test_mask], label_pos[test_mask])

        y_pred = pre_adj[test_mask].cpu().numpy()
        y_true = label_pos[test_mask].cpu().numpy()
        AUC, AUPR, F1, ACC = metrics_graph(y_true, y_pred)

        # Ensure scalars before rounding
        AUC, AUPR, F1, ACC = map(lambda x: round(float(x.mean()), 4), [AUC, AUPR, F1, ACC])
        
        logging.info(f'Test Loss: {round(loss.item(), 4)}')
        logging.info(f'Test AUC: {AUC} | Test AUPR: {AUPR} | Test F1: {F1} | Test ACC: {ACC}')
    return AUC, AUPR, F1, ACC

# Training Loop
best_AUC, best_AUPR, best_F1, best_ACC = 0, 0, 0, 0
for epoch in range(args.epoch):
    logging.info(f'\nEpoch: {epoch}')
    train()
    AUC, AUPR, F1, ACC = test()
    
    if AUC > best_AUC:
        best_AUC, best_AUPR, best_F1, best_ACC = AUC, AUPR, F1, ACC
        torch.save(model.state_dict(), 'best_model.pth')  # Save best model
        logging.info("✅ Best model saved!")

# Final Results
elapsed_time = time.time() - start_time
logging.info('---------------------------------------')
logging.info(f'Elapsed Time: {round(elapsed_time, 4)} seconds')
logging.info(f'Final AUC: {round(best_AUC, 4)} | Final AUPR: {round(best_AUPR, 4)} | Final F1: {round(best_F1, 4)} | Final ACC: {round(best_ACC, 4)}')
logging.info('---------------------------------------')
