import json
import uuid
from pathlib import Path

from huggingface_hub import CommitScheduler

SESSION_ID = str(uuid.uuid4())

feedback_file = Path("user_feedback/") / f"data_{SESSION_ID}.json"
feedback_folder = feedback_file.parent

scheduler = CommitScheduler(
    repo_id="burtenshaw/ohp-test-conversation",
    repo_type="dataset",
    folder_path=feedback_folder,
    path_in_repo="data",
    every=1,
)


def save_feedback(input_object: dict) -> None:
    """
    Append input/outputs and user feedback to a JSON Lines file using a thread lock to avoid concurrent writes from different users.
    """
    with scheduler.lock:
        with feedback_file.open(mode="a") as f:
            input_object["session_id"] = SESSION_ID
            input_object["conversation_id"] = str(uuid.uuid4())
            f.write(json.dumps(obj=input_object))
            f.write("\n")
