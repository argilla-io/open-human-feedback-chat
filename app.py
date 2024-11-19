import gradio as gr
from datasets import Dataset
from pandas import DataFrame

DATASET_REPO_ID = "username/dataset-name"


def add_user_message(history, message):
    """Add a user message to the chat history"""
    if message["text"] is not None:
        content = message["text"]
        history.append(gr.ChatMessage(role="user", content=content))
    return history, gr.Textbox(value=None, interactive=False)


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
            message["liked"] = x.liked
        else:
            message["liked"] = False

        output_data.append(dict(message))

        del message["metadata"]

    return DataFrame(data=output_data)


def submit_conversation(dataframe):
    """ "Submit the conversation to dataset repo"""
    print(dataframe)
    # TODO: Submit the conversation to the dataset repo
    # Dataset.from_pandas(dataframe).push_to_hub(repo_id=DATASET_REPO_ID)


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

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        type="messages",
    )

    chat_input = gr.Textbox(
        interactive=True,
        placeholder="Enter a message...",
        show_label=True,
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
        outputs=[dataframe],
        like_user_message=False,
    )

    gr.Button().click(fn=submit_conversation, inputs=[dataframe], outputs=None)

demo.launch()
