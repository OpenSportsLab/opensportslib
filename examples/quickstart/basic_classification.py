from opensportslib import model


def main():
    """
    Minimal classification example.
    Update config and dataset paths before running.
    """

    my_model = model.classification(
        config="examples/configs/classification_video.yaml"
    )

    my_model.train(
        train_set="/path/to/train_annotations.json",
        valid_set="/path/to/valid_annotations.json",
        pretrained="/path/to/pretrained.pt",  # optional
    )

    metrics = my_model.infer(
        test_set="/path/to/test_annotations.json",
        pretrained="/path/to/checkpoints/best.pt",
        predictions="/path/to/predictions.json",
    )

    print(metrics)


if __name__ == "__main__":
    main()
