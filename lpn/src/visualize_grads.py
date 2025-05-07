import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Tuple, Dict, Any


def trace_recurrent_ga_context(
    model, 
    sample_input: Tuple[jnp.ndarray, jnp.ndarray], 
    matrix_size_rows: int, 
    matrix_size_cols: int, 
    output_dir: str = "gradient_traces"
) -> str:
    """
    Traces the computational graph of the recurrent gradient ascent context computation.
    
    Args:
        model: The LPN model
        sample_input: A tuple of (pairs, grid_shapes) to use for tracing
        matrix_size_rows: Number of rows in the matrix
        matrix_size_cols: Number of columns in the matrix
        output_dir: Directory to save the trace output
        
    Returns:
        Path to the saved trace file
    """
    os.makedirs(output_dir, exist_ok=True)
    trace_file = os.path.join(output_dir, f"rec_ga_trace_{matrix_size_rows}x{matrix_size_cols}.txt")
    
    pairs, grid_shapes = sample_input
    batch_size = pairs.shape[0]
    
    # Get latents from the encoder
    latents_mu, _ = model.encoder(pairs, grid_shapes, dropout_eval=True)
    
    # Function to trace: simplified version of column-wise gradient computation
    def compute_col_gradient(latent_matrix, col_idx):
        # Reshape to create context
        flat_context = latent_matrix.reshape(batch_size, -1)
        flat_context = flat_context[:, None, :]
        
        # Define loss function 
        def compute_loss(latent_matrix):
            total_loss = 0.0
            for i in range(pairs.shape[1]):
                loss, _ = model._loss_from_pair_and_context(
                    context=flat_context, 
                    pairs=pairs[:, i:i+1],
                    grid_shapes=grid_shapes[:, i:i+1],
                    dropout_eval=True,
                    matrix_size_rows=matrix_size_rows,
                    matrix_size_cols=matrix_size_cols,
                )
                total_loss += loss
            return jnp.mean(total_loss)
        
        # Get gradients
        loss, grads = jax.value_and_grad(compute_loss)(latent_matrix)
        
        # Apply column masking
        mask = jnp.arange(matrix_size_cols) == col_idx
        mask = mask.astype(latent_matrix.dtype)
        mask = mask.reshape((1,) * (latent_matrix.ndim - 1) + (-1,))
        masked_grads = grads * mask
        
        return loss, masked_grads
    
    # Create initial latent matrix
    latent_matrix = latents_mu.mean(axis=1).reshape(batch_size, matrix_size_rows, matrix_size_cols)
    
    # Trace the computation for a specific column
    col_idx = 0
    jaxpr = jax.make_jaxpr(compute_col_gradient)(latent_matrix, col_idx)
    
    # Save the JAX program representation
    with open(trace_file, "w") as f:
        f.write(str(jaxpr))
    
    print(f"Computational graph trace saved to {trace_file}")
    return trace_file


def visualize_gradient_computation(
    gradients: List[jnp.ndarray], 
    losses: List[jnp.ndarray], 
    matrix_size_rows: int, 
    matrix_size_cols: int
) -> plt.Figure:
    """
    Visualizes the gradients and losses from recurrent gradient ascent.
    
    Args:
        gradients: List of gradient arrays during optimization
        losses: List of loss values during optimization
        matrix_size_rows: Number of rows in the matrix
        matrix_size_cols: Number of columns in the matrix
        
    Returns:
        Matplotlib figure with visualizations
    """
    # Convert to numpy for easier manipulation
    if not isinstance(gradients[0], np.ndarray):
        gradients = [np.array(g) for g in gradients]
    if not isinstance(losses[0], np.ndarray):
        losses = [np.array(l) for l in losses]
    
    loss_values = np.array(losses)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Plot loss curve
    ax = axes[0, 0]
    ax.plot(loss_values)
    ax.set_title("Loss During Optimization")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    
    # 2. Gradient magnitude per column over time
    col_grads = []
    for g in gradients:
        # Average over batch and rows, get magnitude per column
        col_grad = np.linalg.norm(g, axis=(0, 1))
        col_grads.append(col_grad)
    
    col_grads = np.array(col_grads)  # [steps, cols]
    
    ax = axes[0, 1]
    im = ax.imshow(col_grads, aspect='auto', cmap='viridis')
    ax.set_title("Gradient Magnitude by Column")
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Step")
    plt.colorbar(im, ax=ax)
    
    # 3. Average gradient magnitude across steps
    ax = axes[0, 2]
    avg_col_grads = col_grads.mean(axis=0)
    ax.bar(range(matrix_size_cols), avg_col_grads)
    ax.set_title("Average Gradient Magnitude by Column")
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Average Gradient Magnitude")
    
    # 4. Plot per-column gradient evolution
    ax = axes[1, 0]
    for col_idx in range(min(5, matrix_size_cols)):
        ax.plot(col_grads[:, col_idx], label=f"Col {col_idx}")
    ax.set_title("Gradient Evolution by Column")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Magnitude")
    ax.legend()
    
    # 5. Correlation between columns
    if len(gradients) > 0 and matrix_size_cols > 1:
        ax = axes[1, 1]
        # Take the last gradient measurement
        last_grad = gradients[-1]
        # Reshape to [batch*rows, cols]
        reshaped_grad = last_grad.reshape(-1, matrix_size_cols)
        # Compute correlation between columns
        corr = np.corrcoef(reshaped_grad.T)
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title("Gradient Correlation Between Columns")
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Column Index")
        plt.colorbar(im, ax=ax)
    
    # 6. Simplified computational flow graph
    ax = axes[1, 2]
    ax.axis('off')
    
    # Draw a simple directed graph showing column-wise optimization
    y_levels = np.linspace(0.9, 0.1, matrix_size_cols + 2)
    
    # Draw nodes and connections
    ax.text(0.5, y_levels[0], "Initial Latent", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", fc='lightgray', ec='gray'))
    
    for i in range(matrix_size_cols):
        ax.text(0.5, y_levels[i+1], f"Column {i}", ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc='skyblue', ec='blue'))
        # Arrow from previous to this node
        ax.annotate("", xy=(0.5, y_levels[i+1]), xytext=(0.5, y_levels[i]),
                    arrowprops=dict(arrowstyle="->", color='black'))
    
    ax.text(0.5, y_levels[-1], "Final Context", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc='lightgreen', ec='green'))
    ax.annotate("", xy=(0.5, y_levels[-1]), xytext=(0.5, y_levels[-2]),
                arrowprops=dict(arrowstyle="->", color='black'))
    
    ax.set_title("Recurrent Gradient Ascent Flow")
    
    plt.tight_layout()
    plt.savefig(f"recurrent_ga_flow_{matrix_size_rows}x{matrix_size_cols}.png", dpi=300, bbox_inches='tight')
    
    return fig
  
def visualize_training_gradients(grads, step):
    """
    Visualize gradients during training.
    
    Args:
        grads: Gradient dict from training
        step: Current training step
        
    Returns:
        Matplotlib figure with visualizations
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Gradient norms by layer
    layer_norms = {}
    for module in ["encoder", "decoder"]:
        for key, g in jax.tree_util.tree_flatten_with_path(grads[module])[0]:
            if hasattr(g, "shape"):
                layer_name = "/".join(str(k) for k in key)
                layer_norms[f"{module}/{layer_name}"] = np.linalg.norm(np.array(g))
    
    # Sort layers by norm
    sorted_layers = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)
    
    # Display top 20 layers by gradient magnitude
    top_layers = sorted_layers[:20]
    ax = axes[0, 0]
    ax.barh([layer[0] for layer in top_layers], [layer[1] for layer in top_layers])
    ax.set_title(f"Top 20 Layer Gradient Norms (Step {step})")
    ax.set_xlabel("Gradient Norm")
    
    # 2. Distribution of gradient values
    all_grads = []
    for g in jax.tree_util.tree_leaves(grads):
        if hasattr(g, "shape"):
            all_grads.append(np.array(g).flatten())
    
    all_grads = np.concatenate(all_grads)
    
    ax = axes[0, 1]
    ax.hist(all_grads, bins=50)
    ax.set_title(f"Gradient Distribution (Step {step})")
    ax.set_xlabel("Gradient Value")
    ax.set_ylabel("Count")
    
    # 3. Gradient norm ratio between encoder and decoder
    encoder_norm = optax.global_norm(grads["encoder"])
    decoder_norm = optax.global_norm(grads["decoder"])
    total_norm = optax.global_norm(grads)
    
    ax = axes[1, 0]
    ax.bar(["Encoder", "Decoder", "Total"], [encoder_norm, decoder_norm, total_norm])
    ax.set_title(f"Gradient Norm by Module (Step {step})")
    ax.set_ylabel("Gradient Norm")
    
    # 4. Positive vs Negative gradient ratios
    pos_count = np.sum(all_grads > 0)
    neg_count = np.sum(all_grads < 0)
    zero_count = np.sum(all_grads == 0)
    total_count = len(all_grads)
    
    ax = axes[1, 1]
    ax.pie(
        [pos_count, neg_count, zero_count], 
        labels=["Positive", "Negative", "Zero"],
        autopct='%1.1f%%',
        colors=['green', 'red', 'gray']
    )
    ax.set_title(f"Gradient Direction Distribution (Step {step})")
    
    plt.tight_layout()
    
    return fig