import torch
from torch_geometric.data import Data
from feature_extraction import get_dataframe
import numpy as np
import pandas as pd

def construct_receiver_graph(processed_df, timestamp, rcvId):
    """
    Receiver-centric graph construction matching Algorithm 1
    If no neighbors: single receiver node (index 0)
    """
    # Step 1: Get neighbors N = senders in [t_rcv-1, t_rcv)
    mask = (
    (processed_df['receiver'] == rcvId) &
    (processed_df['rcvTime'] >= timestamp - 1) &
    (processed_df['rcvTime'] < timestamp)
)
    neighbors_df = processed_df[
    mask & (processed_df['sender'] != rcvId)
    ]['sender'].unique()
    N = list(neighbors_df)
    
    # Step 2: Initialize
    node_features = []
    edge_index = []
    edge_attr = []
    
    # ALWAYS include receiver as node 0
    rcv_feat_mask = (processed_df['sender'] == rcvId) & (processed_df['rcvTime'] <= timestamp)
    rcv_latest = processed_df[rcv_feat_mask]
    
    if len(rcv_latest) == 0:
        return None  # No receiver data available
        
    rcv_feat = rcv_latest.iloc[-1]
    feat_rcv = np.array([
        rcv_feat['avg_sender_rate'],    # B
        rcv_feat['speed'],              # S  
        rcv_feat['acceleration'],       # X
        rcv_feat['distance_diff'],      # δ
        rcv_feat['avg_speed_1s']        # μ
    ])
    node_features.append(feat_rcv)
    
    # Step 3: If neighbors exist, add them
    if len(N) > 0:
        node_idx = 1  # Receiver is 0, neighbors start at 1
        
        for neigh_j in N:
            # Get latest neighbor j features at t_rcv  
            j_feat_mask = (processed_df['sender'] == neigh_j) & (processed_df['rcvTime'] <= timestamp)
            j_latest = processed_df[j_feat_mask].iloc[-1]
            
            # Neighbor features (μ from receiver)
            feat_j = np.array([
                j_latest['avg_sender_rate'],      # B
                j_latest['speed'],                # S
                j_latest['acceleration'],         # X  
                j_latest['distance_diff'],        # δ
                rcv_feat['avg_speed_1s']          # μ (receiver's neighbor avg)
            ])
            
            node_features.append(feat_j)
            
            # Bidirectional edges: [0, nodeIdx_j] and [nodeIdx_j, 0]
            nodeIdx_j = node_idx
            edge_index.extend([[0, nodeIdx_j], [nodeIdx_j, 0]])
            node_idx += 1
            
            # Edge attr: placeholder distance
            d = 0.0  # TODO: compute Euclidean from pos_x, pos_y
            edge_attr.extend([[d], [d]])
    
    # Convert to PyG Data object
    x = torch.tensor(node_features, dtype=torch.float)
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
