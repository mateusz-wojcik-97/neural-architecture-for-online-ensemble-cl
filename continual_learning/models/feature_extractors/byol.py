# Code from https://github.com/yaox12/BYOL-PyTorch/blob/master/utils/load_and_convert.py
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from avalanche.benchmarks import SplitCIFAR10, SplitCIFAR100
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from continual_learning.models.feature_extractors.base import FeatureExtractor
from settings import STORAGE_DIR


class ResNet(torch.nn.Module):
    def __init__(self, net_name, pretrained=False, use_fc=False):
        super().__init__()
        base_model = models.__dict__[net_name](pretrained=pretrained)
        self.encoder = torch.nn.Sequential(*list(base_model.children())[:-1])

        self.use_fc = use_fc
        if self.use_fc:
            self.fc = torch.nn.Linear(2048, 512)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        if self.use_fc:
            x = self.fc(x)
        return x


def load_byol_resnet_from_checkpoint(model_path: Path, device: str = 'cpu'):
    device = torch.device(device)
    model = ResNet('resnet50', pretrained=False, use_fc=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)['online_backbone']
    state_dict = {}
    length = len(model.encoder.state_dict())
    for name, param in zip(model.encoder.state_dict(), list(checkpoint.values())[:length]):
        state_dict[name] = param
    model.encoder.load_state_dict(state_dict, strict=True)
    model.eval()
    print(next(model.encoder.parameters()).device)
    return model


class ByolFeatureEncoder(FeatureExtractor):
    def __init__(self, model_path: Path, device: str):
        self.model_path = model_path
        self.model = load_byol_resnet_from_checkpoint(model_path, device=device)

    def preprocess(self, x: Any) -> Any:
        return x

    def get_features(self, x: Any) -> torch.Tensor:
        x = self.preprocess(x)
        return self.model(x)


if __name__ == '__main__':
    from torchvision import transforms

    model_path = STORAGE_DIR / 'models' / 'resnet50_byol' / 'resnet50_byol_imagenet2012.pth.tar'

    model = load_byol_resnet_from_checkpoint(model_path)

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
        n_experiences=10,
        train_transform=cifar10_train_transform,
        eval_transform=cifar10_eval_transform,
    )

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    output_path = Path('output')
    output_path.mkdir(parents=True, exist_ok=True)
    import pickle as pkl

    # train_embeddings_list = []
    for exp_num, experience in enumerate(test_stream):
        current = 0
        dataset = experience.dataset
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for x, y, task in tqdm(data_loader, total=len(data_loader)):
            y = torch.LongTensor(y)

            # print(f'Experience {exp_num} {current} {y.item()}')

            embedding = model(x)

            with open(output_path / f'{y.item()}_{exp_num}_{current}.pkl', 'wb') as file:
                pkl.dump(embedding, file)

            # train_embeddings_list.append(embedding)

            current += 1
            # if current > 10000:
            #     break

    # train_embeddings = torch.cat(train_embeddings_list, dim=0)
    # print(train_embeddings.size())

    # tsne = TSNE()
    # transformed = tsne.fit_transform(train_embeddings.detach().numpy())
    # import seaborn as sns
    # sns.scatterplot(x=transformed[:, 0], y=transformed[:, 1], hue=[int(i / 9) for i in range(90)])
    # plt.show()
