import os
import gradio as gr
from dotenv import load_dotenv
from smolagents import LiteLLMModel, ToolCallingAgent, stream_to_gradio
from search import brave_search_tool
from smolagents.default_tools import FinalAnswerTool, UserInputTool, VisitWebpageTool
from localtools import output_save_to_file # Assuming this is a tool instance

# Need imports for the transcription APIs
# These will be imported within the transcribe_audio function to provide helpful error messages
# if the libraries are not installed.
# import google.generativeai as genai # For Gemini transcription
# from huggingface_hub import InferenceClient # For Hugging Face transcription

# Load environment variables from a .env file
load_dotenv()

# Instantiate agent globally for now to simplify Gradio integration.
# In a production application, consider using dependency injection or a class structure
# to manage the agent instance more formally.
model_name = os.getenv("MODEL_NAME", "gemini/gemini-2.0-flash") # Default to gemini if not set

# Determine which API key to use based on the model name
# This assumes your GEMINI_API_KEY or HF_TOKEN is set in your environment or .env file
api_key = os.getenv("GEMINI_API_KEY") if 'gemini' in model_name.lower() else os.getenv("HF_TOKEN")

# Handle potential missing API keys by printing a warning
if not api_key:
    required_key = 'GEMINI_API_KEY' if 'gemini' in model_name.lower() else 'HF_TOKEN'
    print(f"Warning: {required_key} environment variable not set. Agent functionality requiring this API may be limited.")
    # Note: The code will proceed, but API calls within the agent or transcription might fail.

# Instantiate the model and agent with the determined API key
# Ensure your tool instances are correctly imported and included in the list
model = LiteLLMModel(model_id=model_name, api_key=api_key)
agent = ToolCallingAgent(tools=[brave_search_tool,
                               FinalAnswerTool(), # Instantiate the tool class
                               UserInputTool(),   # Instantiate the tool class
                               VisitWebpageTool(), # Instantiate the tool class
                               output_save_to_file # Assuming this is a pre-instantiated tool object/function
                               ], model=model, add_base_tools=False)


# Function to transcribe audio using either Gemini or Hugging Face
def transcribe_audio(audio_filepath):
    """
    Transcribes the given audio file using either the Gemini API or Hugging Face's
    automatic speech recognition model based on the MODEL_NAME environment variable.
    """
    if audio_filepath is None:
        return "" # Return empty string if no audio file is provided

    transcription = ""
    error_message = ""

    # Check which model to use based on MODEL_NAME
    if 'gemini' in os.getenv("MODEL_NAME", "").lower():
        # Use Gemini API for transcription
        try:
            from google import genai
            # time is imported globally now

            # Ensure GEMINI_API_KEY is set
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                 return "Error: GEMINI_API_KEY not set for Gemini transcription. Please set the environment variable."

            client= genai.Client(api_key= gemini_api_key)

            # Define the prompt specifically for audio transcription with Gemini
            transcription_prompt = 'Generate a transcript of the speech.'

            try:
                # Upload the audio file to Gemini. The SDK handles reading the local file.
                # Based on the user's example and common patterns, assuming genai.upload_file exists.
                print(f"Attempting Gemini transcription - Uploading file: {audio_filepath}")
                file = client.files.upload(file=audio_filepath)
                print(f"File uploaded to Gemini: {file.name}, State: {file.state}")

                response = client.models.generate_content(model= model_name, contents= [transcription_prompt, file])
                transcription = response.text
                print("Gemini transcription successful.")

                # Clean up the uploaded file from Gemini's storage
                try:
                    print(f"Deleting uploaded file from Gemini: {file.name}")
                    if file.name is not None:
                        client.files.delete(name= file.name)
                    print("File deleted from Gemini.")
                except Exception as cleanup_e:
                    # Log cleanup errors but don't prevent returning transcription
                    print(f"Warning: Error deleting Gemini file {file.name}: {cleanup_e}")

            except Exception as e:
                # Catch exceptions during the Gemini transcription process
                error_message = f"Error during Gemini transcription process: {e}"
                print(error_message)

        except ImportError:
            # Handle cases where the google-generativeai library is not installed
            error_message = "Error: google-generativeai library not installed for Gemini transcription. Please install it (`pip install google-generativeai`)."
            print(error_message)
        except Exception as e:
            # Catch any other unexpected errors during Gemini setup
            error_message = f"Unexpected error during Gemini transcription setup: {e}"
            print(error_message)

    else:
        # Use Hugging Face for transcription with the whisper-large-v3 model via FAL AI
        try:
            from huggingface_hub import InferenceClient
            # Ensure HF_TOKEN is set for Hugging Face Inference API
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                return "Error: HF_TOKEN environment variable not set for Hugging Face transcription. Please set it."

            # Instantiate the InferenceClient with the FAL AI provider and API key
            client = InferenceClient(provider="fal-ai", api_key=hf_token)
            print(f"Attempting Hugging Face transcription for: {audio_filepath}")
            # Call the automatic_speech_recognition method, passing the audio file path
            # Assuming this method is awaitable or the client handles async internally.
            output = client.automatic_speech_recognition(audio_filepath, model="openai/whisper-large-v3")
            print(f"Hugging Face transcription raw output: {output}")
            # The output is expected to be a dictionary with a 'text' key.
            transcription = output.get("text", "Transcription failed.")

            # Check if the output indicates an error
            if transcription == "Transcription failed." and "error" in output:
                 error_message = f"Hugging Face API error: {output['error']}"
                 print(error_message)
            elif transcription == "Transcription failed.":
                 error_message = "Hugging Face transcription returned no text."
                 print(error_message)

        except ImportError:
            # Handle cases where the huggingface_hub library is not installed
            error_message = "Error: huggingface_hub library not installed for Hugging Face transcription. Please install it (`pip install huggingface-hub`)."
            print(error_message)
        except Exception as e:
             # Catch any other errors during Hugging Face transcription
             error_message = f"Error during Hugging Face transcription: {e}"
             print(error_message)

    # Return the successful transcription or the captured error message
    return transcription if not error_message else error_message




# This function receives the user's message (text), the chat history, and the audio file path.
def predict(message, history, audio_filepath):
    """
    Processes user input (text or audio), transcribes audio if provided,
    and passes the resulting text input to the agent for processing.
    """
    print(f"Predict function received - Message: '{message}', Audio: {audio_filepath is not None}'")

    user_input = message

    # Handle audio transcription
    if audio_filepath is not None:
        print(f"Audio file received: {audio_filepath}. Starting transcription...")
        transcribed_text = transcribe_audio(audio_filepath)

        if transcribed_text and "Error:" in transcribed_text:
            print(f"Transcription failed with error: {transcribed_text}")
            # Return an error message directly.
            yield "🎙️ (Failed) " + transcribed_text # Yield to update frontend
            return

        print(f"Transcription successful. Transcribed text: '{transcribed_text}'")
        user_input = transcribed_text

    # Handle empty input
    if not user_input or user_input.strip() == "":
        print("No valid input received. Returning current history.")
        yield "" # Yield an empty string to clear the bot's temporary message if any
        return

    print(f"Feeding input to agent: '{user_input}'")

    # Call the streaming function.
    # The 'yield from' construct is used to yield all values from another generator.
    try:
        yield from stream_to_gradio(agent= agent, task= user_input)
        print("Agent processing complete.")

    except Exception as e:
        error_message = f"An error occurred during agent processing: {e}"
        print(error_message)
        yield error_message # Yield the error message to the frontend


# Main function to set up and launch the Gradio interface
def main():
    # The agent and model are instantiated globally above.

    # Set up the Gradio ChatInterface
    # fn: The function that will be called when a new message is submitted (our predict function).
    # additional_inputs: A list of Gradio components to display below the main chat input box.
    # We add an Audio component configured to record from the microphone and return the audio file path.
    iface = gr.ChatInterface(
        fn=predict, # Link the chat interface to our async predict function
        additional_inputs=[
            # Add an Audio component for voice input
            gr.Audio(
                sources=["microphone"], # Allow recording from the microphone
                type="filepath",        # Return the path to a temporary audio file
                label="Voice Input"     # Label for the audio input component in the UI
            )
            # If you wanted other inputs, add them here, e.g., gr.Textbox(label="Extra Info")
        ],
        title="CTIX Analysis Agent Chat", # Title for the Gradio app
        description="Chat with the CTIX Analysis Agent using text or voice.", # Description below the title
        # live=True # Uncomment this line if you want the predict function to be called on every input change (less common for chat)
    )

    # Launch the Gradio interface
    print("Launching Gradio interface...")
    # The launch method starts the web server for the Gradio app.
    # iface.launch(share= True)
    iface.launch(server_name= "0.0.0.0", server_port= 7860)
        # , ssl_keyfile= "./key.pem", ssl_certfile= "./cert.pem")


if __name__ == "__main__":
    # When the script is run directly, call the main function.
    main()
