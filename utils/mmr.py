"""
MMR utilities for diverse document retrieval.
Compatible with all ChromaDB versions.
"""

import numpy as np
from typing import List
from llama_index.core.schema import NodeWithScore


def calculate_mmr_score(
    query_embedding: List[float],
    doc_embedding: List[float],
    selected_embeddings: List[List[float]],
    lambda_param: float = 0.7
) -> float:
    """
    Calculate MMR score for a document.
    
    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    
    Args:
        query_embedding: Query embedding vector
        doc_embedding: Document embedding vector
        selected_embeddings: Embeddings of already selected documents
        lambda_param: Trade-off between relevance (high λ) and diversity (low λ)
                     Default 0.7 = 70% relevance, 30% diversity
    
    Returns:
        MMR score for the document
    """
    # Calculate similarity with query (relevance)
    query_sim = np.dot(query_embedding, doc_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
    )
    
    # Calculate max similarity with already selected docs (redundancy)
    if not selected_embeddings:
        max_selected_sim = 0
    else:
        selected_sims = [
            np.dot(selected_emb, doc_embedding) / (
                np.linalg.norm(selected_emb) * np.linalg.norm(doc_embedding)
            )
            for selected_emb in selected_embeddings
        ]
        max_selected_sim = max(selected_sims)
    
    # MMR formula
    mmr_score = lambda_param * query_sim - (1 - lambda_param) * max_selected_sim
    return mmr_score


def apply_mmr_reranking(
    nodes: List[NodeWithScore],
    query_embedding: List[float],
    top_k: int = 3,
    lambda_param: float = 0.7
) -> List[NodeWithScore]:
    """
    Apply MMR reranking to retrieved nodes for diversity.
    
    Args:
        nodes: List of retrieved nodes with scores
        query_embedding: Query embedding vector
        top_k: Number of documents to return
        lambda_param: MMR lambda parameter (0.7 = 70% relevance, 30% diversity)
    
    Returns:
        Reranked list of nodes using MMR
    """
    if len(nodes) <= top_k:
        return nodes
    
    # Extract embeddings from nodes
    node_embeddings = []
    for node in nodes:
        if hasattr(node.node, 'embedding') and node.node.embedding:
            node_embeddings.append(node.node.embedding)
        else:
            # If no embedding, use zeros (will be ranked low)
            node_embeddings.append([0.0] * len(query_embedding))
    
    # MMR selection
    selected_indices = []
    selected_embeddings = []
    remaining_indices = list(range(len(nodes)))
    
    for _ in range(min(top_k, len(nodes))):
        # Calculate MMR scores for remaining documents
        mmr_scores = []
        for idx in remaining_indices:
            score = calculate_mmr_score(
                query_embedding,
                node_embeddings[idx],
                selected_embeddings,
                lambda_param
            )
            mmr_scores.append((idx, score))
        
        # Select document with highest MMR score
        best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
        
        selected_indices.append(best_idx)
        selected_embeddings.append(node_embeddings[best_idx])
        remaining_indices.remove(best_idx)
    
    # Return selected nodes in MMR order
    return [nodes[idx] for idx in selected_indices]
