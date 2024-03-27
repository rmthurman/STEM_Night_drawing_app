import os
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
import json
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# This example requires environment variables named "OPEN_AI_KEY", "OPEN_AI_ENDPOINT" and "OPEN_AI_DEPLOYMENT_NAME"
# Your endpoint should look like the following https://YOUR_OPEN_AI_RESOURCE_NAME.openai.azure.com/

#retrieve the key and endpoint from the dotenv file

load_dotenv()
OPEN_AI_KEY=os.getenv("OPEN_AI_KEY")
OPEN_AI_ENDPOINT=os.getenv("OPEN_AI_ENDPOINT")
OPEN_AI_DEPLOYMENT_NAME=os.getenv("OPEN_AI_DEPLOYMENT_NAME")
IMAGE_OPEN_AI_KEY=os.getenv("IMAGE_OPEN_AI_KEY")
IMAGE_OPEN_AI_ENDPOINT=os.getenv("IMAGE_OPEN_AI_ENDPOINT")

SPEECH_KEY=os.getenv("SPEECH_KEY")
SPEECH_REGION=os.getenv("SPEECH_REGION")

client = AzureOpenAI(
azure_endpoint=OPEN_AI_ENDPOINT,
api_key=OPEN_AI_KEY,
api_version="2023-12-01-preview"
)

# This will correspond to the custom name you chose for your deployment when you deployed a model.
deployment_id=OPEN_AI_DEPLOYMENT_NAME

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

# Should be the locale for the speaker's language.
speech_config.speech_recognition_language="en-US"
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# The language of the voice that responds on behalf of Azure OpenAI.
speech_config.speech_synthesis_voice_name='en-US-JennyMultilingualNeural'
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)
# tts sentence end mark
tts_sentence_end = [ ".", "!", "?", ";", "。", "！", "？", "；", "\n" ]

# Prompts Azure OpenAI with a request and synthesizes the response.

def draw_image_with_openai(prompt):
    # Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later

    #chat bot tells the user that it is drawing an image based on the prompt
    print(f"Drawing an image: {prompt}")
    speech_synthesizer.speak_text_async("Drawing image: " + prompt + ", please wait.")

    client = AzureOpenAI(
        api_version="2024-02-01",
        azure_endpoint=IMAGE_OPEN_AI_ENDPOINT,
        api_key=IMAGE_OPEN_AI_KEY,
    )

    try:
        result = client.images.generate(
            model="dall-e-3", # the name of your DALL-E 3 deployment
            prompt=prompt, #A polar bear, synthwave style, digital painting
            n=1
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        speech_synthesizer.speak_text_async("I am sorry but I cannot draw: " + prompt)
        return "An error occurred while generating the image."

    image_url = json.loads(result.model_dump_json())['data'][0]['url']

    print(f"Image URL: {image_url}")

    #display the image
    response = requests.get(image_url)
    speech_synthesizer.speak_text_async("Your picture is ready. Here it is.")
    img = Image.open(BytesIO(response.content))
    img.show()

def ask_openai(prompt):
    # Ask Azure OpenAI in streaming way

    #add function to draw image with openai
    tools = [
        {
            "type": "function",
            "function": {
                "name": "draw_image_with_openai",
                "description": "Draws the image based on the prompt",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to generate the image",
                        }
                    },
                    "required": ["prompt"],
                },
            },
        }
    ]


    messages=[{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=deployment_id, max_tokens=200, stream=False, messages=messages,
    tools=tools,
    tool_choice="auto")

    collected_messages = []
    last_tts_request = None

    response_message = response.choices[0].message
    #we can inspect for filtering here

    tool_calls = response_message.tool_calls
    
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "draw_image_with_openai": draw_image_with_openai
        }  # only one function in this example, but you can have multiple
        
        #collected_messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                prompt=function_args.get("prompt")
            )  #The function call has already been made by this point

            collected_messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response, 400 Error is here
    else:
        collected_messages.append(response_message)
        #speak the response message
        speech_synthesizer.speak_text(response_message.content)

# Continuously listens for speech input to recognize and send as text to Azure OpenAI
def chat_with_open_ai():
    
    while True: 
        print("Azure OpenAI is listening. Say 'Stop' or press Ctrl-Z to end the conversation.")
        try:
            # Get audio from the microphone and then send it to the TTS service, max 15 seconds
            speech_recognition_result = speech_recognizer.recognize_once_async().get() #be quiet for 5 secs

            # If speech is recognized, send it to Azure OpenAI and listen for the response.
            if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                if speech_recognition_result.text == "Stop.": 
                    print("Conversation ended.")
                    speech_synthesizer.speak_text("Thank you for using the chatbot. Goodbye.")
                    break
                
                print("Recognized speech: {}".format(speech_recognition_result.text))
                ask_openai(speech_recognition_result.text)
                
        except EOFError:
            break

# Main
try:
    chat_with_open_ai()
except Exception as err:
    print("Encountered exception. {}".format(err))