import uuid
from datetime import datetime

import gradio as gr
from pandas import DataFrame

from feedback import save_feedback


def add_user_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def respond_system_message(history: list):
    """Respond to the user message with a system message"""

    ##############################
    # FAKE RESPONSE
    response = "**That's cool!**"
    ##############################

    # TODO: Add a response to the user message

    message = gr.ChatMessage(role="assistant", content=response)
    history.append(message)
    return history


def wrangle_like_data(x: gr.LikeData, history) -> DataFrame:
    """Wrangle conversations and liked data into a DataFrame"""

    liked_index = x.index[0]

    output_data = []
    for idx, message in enumerate(history):
        if idx == liked_index:
            message["metadata"] = {"title": "liked"}

        liked = True if message["metadata"].get("title") == "liked" else False
        message["liked"] = liked

        output_data.append(
            dict([(k, v) for k, v in message.items() if k != "metadata"])
        )

    return history, DataFrame(data=output_data)


def submit_conversation(dataframe, session_id):
    """ "Submit the conversation to dataset repo"""
    conversation_data = {
        "conversation": dataframe.to_dict(orient="records"),
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "conversation_id": str(uuid.uuid4()),
    }
    save_feedback(input_object=conversation_data)
    gr.Info(f"Submitted {len(dataframe)} messages to the dataset")
    return (gr.Dataframe(value=None, interactive=False), [])


with gr.Blocks(
    css="""
    button[aria-label="dislike"] {
        display: none;
    }
    button[aria-label="like"] {
        width: auto;
    }
    button[aria-label="like"] svg {
        display: none;
    }
    button[aria-label="like"]::before {
        content: "⛔️";
        font-size: 1.5em;
        display: inline-block;
    }
    """
) as demo:
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
