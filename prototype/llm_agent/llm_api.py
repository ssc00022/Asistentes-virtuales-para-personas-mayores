import requests
import os
from requests import Response
from typing import Dict, Any
import json_repair


class LLMApi:
    def __init__(self,
                 api_key: str = os.getenv("<ADA_API_KEY>"),
                 model: str = "<YOUR_MODEL_NAME>"):
        """
        Inicializa la clase LLMApi con los parámetros necesarios para acceder al modelo remoto.

        :param api_key: Clave de acceso a la API (por defecto se toma de las variables de entorno).
        :param model: Ruta o identificador del modelo a usar.
        """
        self.url = "http://<YOUR_URL_SERVER>/v1/chat/completions" # URL del servidor privado
        if not api_key:
            raise ValueError("API Key is required. Set it using the 'api_key' argument or as an environment variable.")
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }

    def send_request(self,
                     chat: list = None,
                     response_format: Dict = None,
                     max_tokens: int = 256,
                     temperature: float = 0.2) -> Response:
        """
        Envía una solicitud POST al servidor Ada para generar una respuesta del modelo.

        :param chat: Historial de mensajes de la conversación.
        :param response_format: Formato de respuesta esperado (JSON Schema, por ejemplo).
        :param max_tokens: Número máximo de tokens en la respuesta generada.
        :param temperature: Nivel de aleatoriedad en la generación de texto.
        :return: Objeto de respuesta HTTP.
        """
        payload = {
            "model": self.model,
            "messages": chat or [],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        if response_format:
            payload["response_format"] = response_format

        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()  # Lanza excepción si la respuesta es un error
        return response

    def invoke(self,
               chat: list = None,
               response_format: Dict = None,
               max_tokens: int = 256,
               temperature: float = 0.2) -> Any:
        """
        Método de invocación que devuelve directamente el contenido textual de la respuesta del modelo.

        :param chat: Historial de conversación.
        :param response_format: Formato de respuesta en JSON Schema, si se requiere.
        :param max_tokens: Límite de tokens en la respuesta.
        :param temperature: Temperatura para controlar la aleatoriedad de la salida.
        :return: Contenido textual de la respuesta del modelo.
        """
        response = self.send_request(chat=chat,
                                     response_format=response_format,
                                     max_tokens=max_tokens,
                                     temperature=temperature)
        return json_repair.loads(response.content.decode("utf-8"))['choices'][0]['message']['content']
