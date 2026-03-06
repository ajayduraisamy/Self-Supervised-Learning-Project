from torch.utils.data import random_split

def create_low_label_split(dataset, label_ratio=0.1):

    labeled_size = int(label_ratio * len(dataset))
    unlabeled_size = len(dataset) - labeled_size

    labeled_data, unlabeled_data = random_split(
        dataset,
        [labeled_size, unlabeled_size]
    )

    return labeled_data, unlabeled_data