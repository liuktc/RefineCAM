from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from torchvision.datasets.folder import default_loader, find_classes, make_dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets import VisionDataset


class Imagewoof(VisionDataset):
    """`Imagewoof <https://github.com/fastai/imagenette#imagenette-1>`_ image classification dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of the Imagenette dataset.
        split (string, optional): The dataset split. Supports ``"train"`` (default), and ``"val"``.
        size (string, optional): The image size. Supports ``"full"`` (default), ``"320px"``, and ``"160px"``.
        download (bool, optional): If ``True``, downloads the dataset components and places them in ``root``. Already
            downloaded archives are not downloaded again.
        transform (callable, optional): A function/transform that takes in a PIL image or torch.Tensor, depends on the given loader,
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        loader (callable, optional): A function to load an image given its path.
            By default, it uses PIL as its image loader, but users could also pass in
            ``torchvision.io.decode_image`` for decoding image data into tensors directly.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class name, class index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (WordNet ID, class index).
    """

    _ARCHIVES = {
        "full": (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz",
            "9aafe18bcdb1632c4249a76c458465ba",
        ),
        "320px": (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz",
            "0f46d997ec2264e97609196c95897a44",
        ),
        "160px": (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz",
            "3d200a7be99704a0d7509be2a9fbfe15",
        ),
    }
    _WNID_TO_CLASS = {
        "n02086240": ("Shih-Tzu"),
        "n02087394": ("Rhodesian_ridgeback"),
        "n02088364": ("beagle"),
        "n02089973": ("English_foxhound"),
        "n02093754": ("Border_terrier"),
        "n02096294": ("Australian_terrier"),
        "n02099601": ("golden_retriever"),
        "n02105641": ("Old_English_sheepdog", "bobtail"),
        "n02111889": ("Samoyed", "Samoyede"),
        "n02115641": ("dingo", "warrigal", "warragal", "Canis_dingo"),
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        size: str = "full",
        download=False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ["train", "val"])
        self._size = verify_str_arg(size, "size", ["full", "320px", "160px"])

        self._url, self._md5 = self._ARCHIVES[self._size]
        self._size_root = Path(self.root) / Path(self._url).stem
        print(self._size_root)
        self._image_root = str(self._size_root / self._split)

        if download:
            self._download()
        elif not self._check_exists():
            print(self._size_root)
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it."
            )

        self.wnids, self.wnid_to_idx = find_classes(self._image_root)
        self.classes = [self._WNID_TO_CLASS[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            class_name: idx
            for wnid, idx in self.wnid_to_idx.items()
            for class_name in self._WNID_TO_CLASS[wnid]
        }
        self._samples = make_dataset(
            self._image_root, self.wnid_to_idx, extensions=".jpeg"
        )
        self.loader = loader

    def _check_exists(self) -> bool:
        return self._size_root.exists()

    def _download(self):
        if self._check_exists():
            return

        download_and_extract_archive(self._url, self.root, md5=self._md5)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        path, label = self._samples[idx]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._samples)
