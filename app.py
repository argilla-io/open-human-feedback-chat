import os
import uuid
from base64 import b64encode
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

import gradio as gr
from huggingface_hub import InferenceClient
from pandas import DataFrame

from feedback import save_feedback

MODEL = os.getenv("MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")
BASE_URL = os.getenv("BASE_URL")
BASE_URL = BASE_URL if BASE_URL else None

client = InferenceClient(
    token=os.getenv("HF_TOKEN"),
    model=(
        os.getenv("MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")
        if BASE_URL is None
        else None
    ),
    base_url=os.getenv("BASE_URL"),
)


def add_user_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def _format_history_as_messages(history: list):
    messages = []
    current_role = None
    current_message_content = []

    for entry in history:
        content = entry["content"]

        if entry["role"] != current_role:
            if current_role is not None:
                messages.append(
                    {"role": current_role, "content": current_message_content}
                )
            current_role = entry["role"]
            current_message_content = []

        if isinstance(content, tuple):  # Handle file paths
            for path in content:
                data_uri = _convert_path_to_data_uri(path)
                current_message_content.append(
                    {"type": "image_url", "image_url": {"url": data_uri}}
                )
        elif isinstance(content, str):  # Handle text
            current_message_content.append({"type": "text", "text": content})

    if current_role is not None:
        messages.append({"role": current_role, "content": current_message_content})

    return messages


def _convert_path_to_data_uri(path) -> str:
    mime_type, _ = guess_type(path)
    with open(path, "rb") as image_file:
        data = image_file.read()
        data_uri = f"data:{mime_type};base64," + b64encode(data).decode("utf-8")
    return data_uri


def _is_file_safe(path) -> bool:
    try:
        return Path(path).is_file()
    except Exception:
        return False


def _process_content(content) -> str | list[str]:
    if isinstance(content, str) and _is_file_safe(content):
        return _convert_path_to_data_uri(content)
    elif isinstance(content, list):
        return _convert_path_to_data_uri(content[0])
    return content


def respond_system_message(history: list) -> list:  # -> list:
    """Respond to the user message with a system message"""
    messages = _format_history_as_messages(history)
    response = client.chat.completions.create(
        messages=messages,
        max_tokens=2000,
        stream=False,
    )
    content = response.choices[0].message.content
    # TODO: Add a response to the user message

    message = gr.ChatMessage(role="assistant", content=content)
    history.append(message)
    return history


def wrangle_like_data(x: gr.LikeData, history) -> DataFrame:
    """Wrangle conversations and liked data into a DataFrame"""

    liked_index = x.index[0]

    output_data = []
    for idx, message in enumerate(history):
        if idx == liked_index:
            message["metadata"] = {"title": "liked" if x.liked else "disliked"}
        rating = message["metadata"].get("title")
        if rating == "liked":
            message["rating"] = 1
        elif rating == "disliked":
            message["rating"] = -1
        else:
            message["rating"] = None

        output_data.append(
            dict([(k, v) for k, v in message.items() if k != "metadata"])
        )

    return history, DataFrame(data=output_data)


def submit_conversation(dataframe, session_id):
    """ "Submit the conversation to dataset repo"""
    if dataframe.empty:
        gr.Info("No messages to submit because the conversation was empty")
        return (gr.Dataframe(value=None, interactive=False), [])

    dataframe["content"] = dataframe["content"].apply(_process_content)
    conversation_data = {
        "conversation": dataframe.to_dict(orient="records"),
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "conversation_id": str(uuid.uuid4()),
    }
    save_feedback(input_object=conversation_data)
    gr.Info(f"Submitted {len(dataframe)} messages to the dataset")
    return (gr.Dataframe(value=None, interactive=False), [])


with gr.Blocks() as demo:
    ##############################
    # Chatbot
    ##############################
    session_id = gr.Textbox(
        interactive=False,
        value=str(uuid.uuid4()),
        visible=False,
    )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        type="messages",
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
        submit_btn=True,
    )

    chat_msg = chat_input.submit(
        fn=add_user_message, inputs=[chatbot, chat_input], outputs=[chatbot, chat_input]
    )

    bot_msg = chat_msg.then(
        respond_system_message, chatbot, chatbot, api_name="bot_response"
    )

    bot_msg.then(lambda: gr.Textbox(interactive=True), None, [chat_input])

    ##############################
    # Deal with feedback
    ##############################

    dataframe = gr.DataFrame()

    chatbot.like(
        fn=wrangle_like_data,
        inputs=[chatbot],
        outputs=[chatbot, dataframe],
        like_user_message=False,
    )

    gr.Button(
        value="Submit conversation",
    ).click(
        fn=submit_conversation,
        inputs=[dataframe, session_id],
        outputs=[dataframe, chatbot],
    )
    demo.load(
        lambda: str(uuid.uuid4()),
        inputs=[],
        outputs=[session_id],
    )

demo.launch()
