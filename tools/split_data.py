from numpy.random import default_rng
from pathlib import Path
import numpy as np
from cv2 import imwrite, imread
from sklearn.model_selection import train_test_split


def read_images(base_dir: Path, categories: "list[str]"):
    samplesDict = {category: [] for category in categories}
    for category in categories:
        print("Loading:", category)
        dir_path = base_dir / category
        images = [
            imread(str(path.absolute())) for path in list(dir_path.iterdir())
        ]
        samplesDict[category].extend(images)
    print("Read all images")
    print("--------------------------")
    return samplesDict


def split_data(samples_dict: dict, categories, split=[0.65, 0.15, 0.20]):
    split_sets = {}
    for idx, cat in enumerate(categories):
        print("Splitting dateset category", cat)
        cat_samples = samples_dict[cat]
        cat_labels = np.full(len(cat_samples), 1)
        split_sets[cat] = split_images(cat_samples, cat_labels, split)
    print("Completed splitting")
    print("--------------------------")
    return split_sets


def split_images(samples, labels, split):
    """Splits the data randomly into the given split and returns
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    train, validation, test = split
    if sum(split) != 1:
        raise Exception("Invalid split - must be equal to 1")

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(samples), labels, test_size=test, random_state=45
    )
    if not validation:
        return X_train, None, X_test, y_train, None, y_test

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=(validation / train), random_state=45
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def write_all(sets_dict: dict, base_dir, categories: "list[str]"):
    for cat in categories:
        sample_sets = sets_dict[cat][:3]
        write_images(sample_sets, base_dir, categories)
    print("Written all images")
    print("--------------------------")


def write_images(sample_sets: list, base_dir: Path, categories: "list[str]"):
    (
        X_train,
        X_val,
        X_test,
    ) = sample_sets
    for set_type in ["train", "validation", "test"]:
        print("Writing Set", set_type)
        if set_type == "train":
            write_dataset_images(X_train, base_dir / set_type, categories)
        elif set_type == "validation":
            write_dataset_images(X_val, base_dir / set_type, categories)
        else:
            write_dataset_images(X_test, base_dir / set_type, categories)


def write_dataset_images(images, set_path: Path, categories: "list[str]"):
    for category in categories:
        print("Writing category", category)
        write_category_of_imgs(images, set_path, category)


def write_category_of_imgs(images, dir_path: Path, category: str):
    cat_path = dir_path / category
    Path(cat_path).mkdir(parents=True, exist_ok=True)

    saved_samples = list(
        map(
            lambda path: int(str(path).split("_")[-1].split(".")[0]),
            list(cat_path.iterdir()),
        )
    )
    if saved_samples:
        last_sample_added = np.max(saved_samples) + 1
    else:
        last_sample_added = 1

    for idx, image in enumerate(images, start=last_sample_added):
        imwrite(str((cat_path / f"{category}_{idx}.png").absolute()), image)


if __name__ == "__main__":
    base_path_in = Path("data/processed/crop_6_1000x1000/")
    categories = ["dc"  , "marvel"]
    base_path_out = Path("data/sets/6_999x999")
    image_dict = read_images(base_path_in, categories)
    splits_dict = split_data(image_dict, categories)
    write_all(splits_dict, base_path_out, categories)
