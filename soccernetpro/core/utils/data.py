from collections import defaultdict
from torch.utils.data import Subset
import torch

def balanced_subset(dataset, samples_per_class=5):
    class_indices = defaultdict(list)

    for idx in range(len(dataset)):
        label = dataset.samples[idx]["label"]
        class_indices[label].append(idx)

    print(class_indices.keys())
    selected_indices = []
    for label, indices in class_indices.items():
        selected_indices.extend(indices[:samples_per_class])

    print(selected_indices)
    return Subset(dataset, selected_indices)


def batch_tensor(tensor, dim=1, squeeze=False):
    """
    A function to reshape PyTorch tensor `tensor` along some dimension `dim` to the batch dimension 0 such that the tensor can be processed in parallel. 
    If `sqeeze`=True, the dimension `dim` will be removed completely, otherwise it will be of size=1. Check `unbatch_tensor()` for the reverese function.
    """
    batch_size, dim_size = tensor.shape[0], tensor.shape[dim]
    returned_size = list(tensor.shape)
    returned_size[0] = batch_size * dim_size
    returned_size[dim] = 1
    if squeeze:
        return tensor.transpose(0, dim).reshape(returned_size).squeeze_(dim)
    else:
        return tensor.transpose(0, dim).reshape(returned_size)


def unbatch_tensor(tensor, batch_size, dim=1, unsqueeze=False):
    """
    A function to chunk pytorch tensor `tensor` along the batch dimension 0 and concatenate the chuncks on dimension `dim` to recover from `batch_tensor()` function.
    If `unsqueee`=True, it will add a dimension `dim` before the unbatching. 
    """
    fake_batch_size = tensor.shape[0]
    nb_chunks = int(fake_batch_size / batch_size)
    if unsqueeze:
        return torch.cat(torch.chunk(tensor.unsqueeze_(dim), nb_chunks, dim=0), dim=dim).contiguous()
    else:
        return torch.cat(torch.chunk(tensor, nb_chunks, dim=0), dim=dim).contiguous()


from torch_geometric.data import Batch

def tracking_collate_fn(batch):
    """
    Custom collate function for tracking data.
    Uses PyG Batch.from_data_list for efficient C++ batching.
    """
    batch_size = len(batch)
    seq_len = batch[0]['seq_len']
    
    # flatten all graphs from all samples
    all_graphs = []
    for sample_idx, item in enumerate(batch):
        for time_idx, graph in enumerate(item['graphs']):
            all_graphs.append(graph)
    
    # PyG handles node offsets for edge_index automatically
    batched_graphs = Batch.from_data_list(all_graphs)
    
    return {
        'x': batched_graphs.x,
        'edge_index': batched_graphs.edge_index,
        'batch': batched_graphs.batch,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'labels': torch.tensor([item['label'] for item in batch], dtype=torch.long),
        'id': [item['id'] for item in batch],
    }

def mixup_data(x, y, alpha=0.2):
    """blend pairs of samples and their labels for mixup augmentation."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam
    