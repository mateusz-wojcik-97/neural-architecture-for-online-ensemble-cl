from avalanche.benchmarks import GenericCLScenario, SplitCIFAR10, SplitCIFAR100, SplitMNIST
from omegaconf import DictConfig
from torchvision import transforms


def load_scenario(dataset_name: str, config: DictConfig) -> GenericCLScenario:
    if dataset_name == 'mnist':
        transform_mnist = transforms.Compose([
            transforms.ToTensor(),
        ])
        scenario = SplitMNIST(
            n_experiences=config.num_experiences,
            return_task_id=config.return_task_id,
            fixed_class_order=config.fixed_class_order,
            train_transform=transform_mnist,
            eval_transform=transform_mnist,
            shuffle=config.shuffle,
        )
        return scenario
    elif dataset_name == 'cifar10':
        cifar10_train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            transforms.Resize(size=(256, 256))
        ])
        cifar10_eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            transforms.Resize(size=(256, 256))
        ])
        scenario = SplitCIFAR10(
            n_experiences=config.num_experiences,
            return_task_id=config.return_task_id,
            fixed_class_order=config.fixed_class_order,
            train_transform=cifar10_train_transform,
            eval_transform=cifar10_eval_transform,
        )
        return scenario
    elif dataset_name == 'cifar100':
        cifar100_train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762)),
            transforms.Resize(size=(256, 256))
        ])
        cifar100_eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762)),
            transforms.Resize(size=(256, 256))
        ])
        scenario = SplitCIFAR100(
            n_experiences=config.num_experiences,
            return_task_id=config.return_task_id,
            fixed_class_order=config.fixed_class_order,
            train_transform=cifar100_train_transform,
            eval_transform=cifar100_eval_transform,
        )
        return scenario
    else:
        raise ValueError(f"Dataset {dataset_name} is not available")
