from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


class Pipeline:
    class Valves(BaseModel):
        # You can add your custom valves here.
        AZURE_ML_ENDPOINT: str
        AZURE_ML_KEY: str

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "azure_ml_pipeline"
        self.name = "Azure ML Phi-3-medium-128k"
        self.valves = self.Valves(
            **{
                "AZURE_ML_ENDPOINT": os.getenv("AZURE_ML_ENDPOINT", "your-endpoint-here"),
                "AZURE_ML_KEY": os.getenv("AZURE_ML_KEY", "your-key-here"),
            }
        )
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        client = ChatCompletionsClient(
            endpoint=self.valves.AZURE_ML_ENDPOINT,
            credential=AzureKeyCredential(self.valves.AZURE_ML_KEY)
        )

        response = None
        response = client.complete(
            messages=[
                SystemMessage(messages),
                UserMessage(user_message),
            ]
        )

        return response.choices[0].message.content
