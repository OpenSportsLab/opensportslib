from opensportslib.apis import LocalizationModel


def main():
    """
    Minimal localization example.
    Update config and dataset paths before running.
    """

    my_model = LocalizationModel(
        config="examples/configs/localization.yaml",
        weights="/path/to/weights.pt",  # optional
    )

    my_model.train(
        train_set="/path/to/train_annotations.json",
        valid_set="/path/to/valid_annotations.json",
    )
    
    predictions = my_model.infer(
        test_set="/path/to/test_annotations.json",
    )

    print(predictions)

    metrics = my_model.evaluate(
        test_set="/path/to/test_annotations.json",
    )

    print(metrics)


if __name__ == "__main__":
    main()
