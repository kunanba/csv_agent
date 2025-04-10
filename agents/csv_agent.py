import asyncio
import logging
import os
import streamlit as st

from semantic_kernel.agents import AssistantAgentThread, AzureAssistantAgent
from semantic_kernel.contents import StreamingFileReferenceContent

logging.basicConfig(level=logging.ERROR)

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
csv_file_path_1 = os.path.join(
    parent_dir, "data_processing", "csv_tables", "merged_analysis_of_financial_experience.csv"
)
csv_file_path_2 = os.path.join(
    parent_dir, "data_processing", "csv_tables", "merged_schedules_of_changes_in_net_pension_liability.csv"
)

async def download_file_content(agent: AzureAssistantAgent, file_id: str):
    try:
        response_content = await agent.client.files.content(file_id)
        # Save the file as PNG 
        file_path = os.path.join(os.getcwd(), f"{file_id}.png")
        with open(file_path, "wb") as file:
            file.write(response_content.content)
        st.write(f"File saved to: {file_path}")
        if file_path.lower().endswith(".png"):
            st.image(file_path)
    except Exception as e:
        st.write(f"An error occurred while downloading file {file_id}: {str(e)}")

async def download_response_image(agent: AzureAssistantAgent, file_ids: list[str]):
    if file_ids:
        for file_id in file_ids:
            await download_file_content(agent, file_id)

async def initialize_agent():
    client, model = AzureAssistantAgent.setup_resources()
    file_ids = []
    for path in [csv_file_path_1, csv_file_path_2]:
        with open(path, "rb") as file:
            file_obj = await client.files.create(file=file, purpose="assistants")
            file_ids.append(file_obj.id)

    instructions = """
        Analyze the available data to provide an answer to the user's question.
        Always format your response using markdown and include numerical indexing for lists or tables.
        Always sort lists in ascending order.
        Please use the context of the previous conversation to answer follow-up questions.
        Always include a chart or diagram to illustrate your findings.
    """
    
    code_interpreter_tools, code_interpreter_tool_resources = AzureAssistantAgent.configure_code_interpreter_tool(
        file_ids=file_ids
    )

    definition = await client.beta.assistants.create(
        model=model,
        instructions=instructions,
        name="SampleAssistantAgent",
        tools=code_interpreter_tools,
        tool_resources=code_interpreter_tool_resources,
    )

    agent = AzureAssistantAgent(
        client=client,
        definition=definition,
    )
    return agent, client

async def process_user_input(user_input: str, agent: AzureAssistantAgent, thread: AssistantAgentThread = None):
    collected_responses = ""
    file_ids = []
    is_code = False
    last_role = None

    async for response in agent.invoke_stream(messages=user_input, thread=thread):
        current_is_code = response.metadata.get("code", False)
        content_text = str(response.content)
        
        if current_is_code:
            if not is_code:
                collected_responses += "\n\n```python\n"
                is_code = True
            collected_responses += content_text
        else:
            if is_code:
                collected_responses += "\n```\n"
                is_code = False
                last_role = None
            if hasattr(response, "role") and response.role is not None and last_role != response.role:
                collected_responses += f"\n**{response.role}:** "
                last_role = response.role
            collected_responses += content_text

        file_ids.extend([
            item.file_id for item in response.items if isinstance(item, StreamingFileReferenceContent)
        ])
        thread = response.thread

    if is_code:
        collected_responses += "\n```\n"

    await download_response_image(agent, file_ids)
    return collected_responses, thread


st.title("Azure Assistant Agent Streamlit App")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""
if "agent" not in st.session_state:
    st.info("Initializing agent and uploading files. Please wait...")
    agent, client = asyncio.run(initialize_agent())
    st.session_state.agent = agent
    st.session_state.client = client
    st.session_state.thread = None
    st.success("Agent initialized.")

with st.form(key="chat_form"):
    user_input = st.text_input("Enter your question for the agent (type 'exit' to end):")
    submit_button = st.form_submit_button(label="Submit")

if submit_button and user_input:
    if user_input.lower() == "exit":
        st.info("Exiting conversation. You can now clean up resources.")
    else:
        st.info("Processing your query, please wait...")
        responses, updated_thread = asyncio.run(
            process_user_input(user_input, st.session_state.agent, st.session_state.thread)
        )
        st.session_state.thread = updated_thread
        st.session_state.chat_history += f"\n**User:** {user_input}\n"
        st.session_state.chat_history += responses
        st.markdown(st.session_state.chat_history)

if st.button("Cleanup Resources"):
    async def cleanup():
        if st.session_state.get("thread"):
            await st.session_state.agent.client.beta.assistants.delete(st.session_state.agent.id)
            await st.session_state.thread.delete()
        st.session_state.clear()
    asyncio.run(cleanup())
    st.success("Resources cleaned up.")
