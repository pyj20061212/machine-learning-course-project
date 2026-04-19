import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_digits_data(
    test_size: float = 0.25,
    random_state: int = 42,
    standardize: bool = True,
):
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)
    images = digits.images

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        np.arange(len(X)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    images_train = images[idx_train]
    images_test = images[idx_test]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "images_train": images_train,
        "images_test": images_test,
        "scaler": scaler,
        "raw_X": X,
        "raw_y": y,
        "raw_images": images,
    }