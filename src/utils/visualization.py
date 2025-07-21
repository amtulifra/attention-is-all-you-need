import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from matplotlib import cm

def plot_attention_weights(
    attention_weights: torch.Tensor,
    src_tokens: list[str],
    tgt_tokens: list[str],
    layer: int = 0,
    head: int = 0,
    save_path: str = None,
    show: bool = True,
    figsize: tuple[int, int] = (10, 10),
    cmap: str = 'viridis'
) -> None:
    """Plot attention weights for a specific layer and head."""
    attn = attention_weights[layer][head].detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        attn,
        cmap=cmap,
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        linewidths=0.1,
        linecolor='gray',
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.title(f'Attention Weights (Layer {layer}, Head {head})')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    plt.close()

def plot_attention_heads(
    attention_weights: torch.Tensor,
    src_tokens: list[str],
    tgt_tokens: list[str],
    layer: int = 0,
    n_cols: int = 4,
    save_path: str = None,
    show: bool = True,
    figsize: tuple[int, int] = (20, 15),
    cmap: str = 'viridis'
) -> None:
    """Plot attention weights for all heads in a specific layer."""
    num_heads = attention_weights.size(1)
    n_rows = math.ceil(num_heads / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    for i in range(num_heads):
        row, col = divmod(i, n_cols)
        attn = attention_weights[layer, i].detach().cpu().numpy()
        
        sns.heatmap(
            attn,
            ax=axes[row, col],
            cmap=cmap,
            xticklabels=src_tokens if row == n_rows - 1 else [],
            yticklabels=tgt_tokens if col == 0 else [],
            cbar=False,
            square=True,
            linewidths=0.1,
            linecolor='gray'
        )
        axes[row, col].set_title(f'Head {i}')
    
    # Remove empty subplots
    for i in range(num_heads, n_rows * n_cols):
        fig.delaxes(axes[i // n_cols, i % n_cols])
    
    fig.colorbar(axes[0, 0].collections[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    plt.suptitle(f'Attention Weights (Layer {layer})', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    plt.close()

def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_metrics: dict = None,
    val_metrics: dict = None,
    save_path: str = None,
    show: bool = True,
    figsize: tuple[int, int] = (15, 10)
) -> None:
    """Plot training and validation loss curves and metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    if train_metrics and val_metrics:
        for metric_name in train_metrics:
            if metric_name in val_metrics:
                ax2.plot(epochs, train_metrics[metric_name], 'b--', label=f'Train {metric_name.capitalize()}')
                ax2.plot(epochs, val_metrics[metric_name], 'r--', label=f'Val {metric_name.capitalize()}')
        
        ax2.set_title('Training and Validation Metrics')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Metric Value')
        ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    plt.close()

def plot_embeddings(
    embeddings: torch.Tensor,
    labels: list[str],
    n_samples: int = 1000,
    save_path: str = None,
    show: bool = True,
    figsize: tuple[int, int] = (15, 15)
) -> None:
    """Visualize embeddings using t-SNE."""
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    unique_labels = sorted(set(labels))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))
    
    plt.figure(figsize=figsize)
    for label, color in label_to_color.items():
        mask = np.array(labels) == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            color=color,
            label=label,
            alpha=0.7,
            s=50
        )
    
    plt.title('t-SNE Visualization of Embeddings')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    plt.close()
