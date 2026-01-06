from collections import defaultdict
from torch.utils.data import Subset

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