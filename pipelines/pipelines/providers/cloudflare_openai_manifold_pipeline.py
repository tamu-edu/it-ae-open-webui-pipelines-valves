from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel

import os
import requests


class Pipeline:
    class Valves(BaseModel):
        CLOUDFLARE_OPENAI_API_BASE_URL: str = ""
        CLOUDFLARE_OPENAI_API_KEY: str = ""
        CLOUDFLARE_ACCOUNT_ID: str = ""
        CLOUDFLARE_AI_GATEWAY_ID: str = ""
        pass

    def __init__(self):
        self.type = "manifold"
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "cloudflare_openai_pipeline"
        self.name = "Cloudflare (OpenAI API): "

        self.valves = self.Valves(
            **{
                "CLOUDFLARE_OPENAI_API_KEY": os.getenv(
                    "CLOUDFLARE_OPENAI_API_KEY", "your-cloudflare-openai-api-key-here"
                ),
                "CLOUDFLARE_OPENAI_API_BASE_URL": os.getenv(
                    "CLOUDFLARE_OPENAI_API_BASE_URL", "your-cloudflare-openai-api-base-url-here"
                ),
                "CLOUDFLARE_ACCOUNT_ID": os.getenv(
                    "CLOUDFLARE_ACCOUNT_ID", "your-cloudflare-account-id-here"
                ),
                "CLOUDFLARE_AI_GATEWAY_ID": os.getenv(
                    "CLOUDFLARE_AI_GATEWAY_ID", "your-cloudflare-ai-gateway-id-here"
                )
            }
        )

        self.pipelines = self.get_openai_models()
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        print(f"on_valves_updated:{__name__}")
        self.pipelines = self.get_openai_models()
        pass

#    def modify_model_path(self, original_string):
#        parts = original_string.split('/')
#        suffix = '/'.join(parts[2:])
#        print(f"\nDEBUG: {parts}, {suffix}\n\n")
#        return suffix

    def modify_model_path(self, input_string):
        # Remove the '@' symbol
        cleaned_string = input_string.replace('@', '')
        # Replace slashes with underscores
        cleaned_string = cleaned_string.replace('/', '_')
        return cleaned_string

    def reverse_model_path(self, input_string):
        # Replace underscores with slashes
        reversed_string = input_string.replace('_', '/')
        # Add the '@' symbol at the front
        reversed_string = '@' + reversed_string
        return reversed_string

    def get_openai_models(self):
        if self.valves.CLOUDFLARE_OPENAI_API_KEY:
            try:
                headers = {}
                headers["Authorization"] = f"Bearer {self.valves.CLOUDFLARE_OPENAI_API_KEY}"
                headers["Content-Type"] = "application/json"

                print(f"DEBUG:{__name__}")
                print(f"DEBUG: {__name__}, Loaded API Key: {self.valves.CLOUDFLARE_OPENAI_API_KEY}")
                print(f"DEBUG: {__name__}, Base URL: {self.valves.CLOUDFLARE_OPENAI_API_BASE_URL}")
                print(f"DEBUG: {__name__}, Base URL: {self.valves.CLOUDFLARE_ACCOUNT_ID}")
                print(f"DEBUG: {__name__}, Account ID: {self.valves.CLOUDFLARE_ACCOUNT_ID}")
                print(f"DEBUG: {__name__}, Full URL: {self.valves.CLOUDFLARE_OPENAI_API_BASE_URL}/{self.valves.CLOUDFLARE_ACCOUNT_ID}/ai/search")

                r = requests.get(
                    f"{self.valves.CLOUDFLARE_OPENAI_API_BASE_URL}/{self.valves.CLOUDFLARE_ACCOUNT_ID}/ai/models/search", headers=headers
                )

                models = r.json()

                print(f"DEBUG:{__name__}, MODELS: {models}")

                return [
                    {
                        #"id": model["id"],
                        #"name": model["name"] if "name" in model else model["id"],
                        "id": self.modify_model_path(model["name"]) if "name" in model else model["id"],
                        "name": self.modify_model_path(model["name"]) if "name" in model else model["id"],
                    }
                    for model in models["result"]
                    if "@cf" in model["name"]
                ]

            except Exception as e:

                print(f"Error: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from OpenAI, please update the API Key in the valves.",
                    },
                ]
        else:
            return []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        headers = {}
        headers["Authorization"] = f"Bearer {self.valves.CLOUDFLARE_OPENAI_API_KEY}"
        headers["Content-Type"] = "application/json"

        model_id = self.reverse_model_path(model_id)

        payload = {**body, "model": model_id}

        print("----------------------------------------")
        print(f"Payload {__name__}: {payload}")

        if "user" in payload:
            del payload["user"]
        if "chat_id" in payload:
            del payload["chat_id"]
        if "title" in payload:
            del payload["title"]

        print("----------------------------------------")
        print(f"Payload {__name__}: {payload}")
        print(f"URL {__name__}: {self.valves.CLOUDFLARE_OPENAI_API_BASE_URL}/{self.valves.CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions")
        print(f"Headers {__name__}: {headers}")
        print("----------------------------------------")

        try:
            r = requests.post(
                url=f"{self.valves.CLOUDFLARE_OPENAI_API_BASE_URL}/{self.valves.CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"
