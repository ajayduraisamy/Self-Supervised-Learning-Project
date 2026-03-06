import torchvision.transforms as transforms

def ssl_augmentation():

    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.ToTensor()
    ])

    return transform