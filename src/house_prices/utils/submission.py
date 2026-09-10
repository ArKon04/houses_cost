from pathlib import Path

import pandas as pd


def create_submission(sample_submission_path, predictions, output_path, target_column):
    

    submission = pd.read_csv(sample_submission_path)

    if len(submission) != len(predictions):
        raise ValueError(
            "Number of predictions does not match "
            "sample submission size"
        )

    submission[target_column] = predictions

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    submission.to_csv(
        output_path,
        index=False
    )