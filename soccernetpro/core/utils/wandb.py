import wandb
import matplotlib.pyplot as plt
import numpy as np

def log_attention_wandb(attention, split_name):

    attn = attention.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_title(f"{split_name} Attention Map")
    ax.set_xlabel("Views / Time")
    ax.set_ylabel("Batch")

    wandb.log({
        f"{split_name}/attention_map": wandb.Image(fig)
    })

    plt.close(fig)


def log_sample_videos_wandb(mvclips, preds, labels, split_name, max_samples=2, fps=5):


    # mvclips: (B, V, C, T, H, W)
    mvclips = mvclips.detach().cpu().numpy()

    for i in range(min(len(mvclips), max_samples)):
        views = mvclips[i]  # (V, C, T, H, W)

        # Log each view separately
        for v in range(views.shape[0]):
            video = views[v].transpose(1, 2, 3, 0)  # (T, H, W, C)
            video = (video * 255).astype(np.uint8) if video.max() <= 1.0 else video

            wandb.log({
                f"{split_name}/sample_{i}_view_{v}": wandb.Video(
                    video,
                    fps=fps,
                    caption=f"Pred: {preds[i]}, GT: {labels[i]}"
                )
            })


def log_confusion_matrix_wandb(y_true, y_pred, class_names, split_name):
    wandb.log({
        f"{split_name}/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=class_names
        )
    })
