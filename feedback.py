import json
import uuid
from pathlib import Path

from huggingface_hub import CommitScheduler

feedback_file = Path("user_feedback/") / f"data_{uuid.uuid4()}.json"
feedback_folder = feedback_file.parent

scheduler = CommitScheduler(
    repo_id="report-translation-feedback",
    repo_type="dataset",
    folder_path=feedback_folder,
    path_in_repo="data",
    every=10,
)


def save_feedback(input_object: dict) -> None:
    """
    Append input/outputs and user feedback to a JSON Lines file using a thread lock to avoid concurrent writes from different users.
    """
    with scheduler.lock:
        with feedback_file.open("a") as f:
            f.write(json.dumps(obj=input_object))
            f.write("\n")
