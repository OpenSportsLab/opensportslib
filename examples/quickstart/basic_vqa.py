from opensportslib.apis import VQAModel


def main():
    """
    Minimal VQA example.
    Update config, question, and dataset paths before running.
    """

    my_model = VQAModel(
        config="examples/configs/vqa_qwen.yaml",
        weights=None,  # optional: path or Hugging Face model ID
    )

    predictions = my_model.infer(
        test_set="/path/to/test_annotations.json",
    )

    print(predictions)

    single_prediction = my_model.infer(
        video_path="/path/to/video.mp4",
        question="What card would you give? Why?",
    )

    print(single_prediction)

    metrics = my_model.evaluate(
        test_set="/path/to/test_annotations.json",
        predictions=predictions,
    )

    print(metrics)


if __name__ == "__main__":
    main()
