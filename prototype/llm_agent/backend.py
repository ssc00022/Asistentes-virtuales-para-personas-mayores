from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import aiofiles
import os
import asyncio
import whisper
from uuid import uuid4
import edge_tts
import logging
import traceback

from llm_api import LLMApi
from agent import Agent
from tools import retrieval_augmented_generation


class Backend:
    def __init__(self):
        # Instancia principal de FastAPI
        self.local_server = FastAPI()

        # Configura logging para mostrar errores o mensajes informativos en consola
        logging.basicConfig(level=logging.INFO)

        # Carga del modelo de transcripción Whisper en CPU
        try:
            self.transcriptor = whisper.load_model(name="turbo", device="cpu")
        except Exception as e:
            logging.error("Error al cargar el modelo de Whisper: %s", e)

        # Prompt de sistema personalizado para el asistente virtual
        self.system_prompt = (
            "Eres una asistente conversacional llamada María, diseñada especialmente para una persona mayor con el siguiente perfil: "
            "- Nombre y apellidos: {nombre}. "
            "- Edad: {edad}. "
            "- Lugar de nacimiento: {lugarNacimiento}. "
            "- Familiares cercanos: {familiares}. "
            "- Gustos personales: {gustos}. "
            "Debes usar un lenguaje sencillo, breve y educado para comunicarte con el usuario. "
            "- Sencillo: Usa palabras básicas y evita términos complejos o técnicos. "
            "- Breve: Tus respuestas no deben ser demasiado largas, preferiblemente de unas pocas líneas. "
            "- Educado: Debes dirigirte a la persona mayor con cortesía, utilizando el pronombre 'usted' para "
            "referirte al usuario, a menos que conozcas su nombre. "
            "Tu rol es activo e interactivo, lo que significa que debes proponer y mantener conversaciones sobre "
            "temas cotidianos, recuerdos del pasado, gustos personales o estado de los familiares cercanos. "
            "Tu objetivo es hacer que la persona mayor se sienta acompañada y entretenida. "
            "No posees la capacidad de hablar sobre temas actuales como el tiempo que hace ahora, los últimos resultados deportivos o "
            "las últimas noticias en el mundo porque no puedes hacer búsquedas en internet. Sin embargo, tienes "
            "la capacidad de buscar información sobre la Provincia de Jaén como sus pueblos, sus montañas y ríos, "
            "historias, leyendas y personajes célebres, entre otros. "
            "No puedes inventar ninguna información durante la conversación. Si no posees suficiente información "
            "para responder, estrictamente debes indicar que no tienes suficiente información y proponer otro tema "
            "de conversación diferente."
        )

        # Inicialización de variables que se usarán tras el setup
        self.agent = None
        self.config = None

        # Ruta de carpetas temporales para almacenar los audios
        self.upload_folder = "audio_temp"
        self.response_audio_path = os.path.join(self.upload_folder, "response.mp3")
        os.makedirs(self.upload_folder, exist_ok=True)

        # Definición de las rutas de la API
        self.define_routes()

    def define_routes(self):
        # Endpoint raíz que indica si el servidor está operativo
        @self.local_server.get("/")
        async def root():
            return {"message": "Servidor operativo"}

        # Endpoint que inicializa el agente con los datos del usuario
        @self.local_server.post("/setup")
        async def setup(user_info: dict):
            """
            Inicializa el agente con la información personalizada del usuario y genera un mensaje de bienvenida.
            """
            # Rellena el prompt con los datos del usuario
            system_prompt = self.system_prompt.format(**user_info)

            # Inicializa el agente con el prompt y las herramientas
            self.agent = Agent(
                llm=LLMApi(),
                tools=[retrieval_augmented_generation],
                system_prompt=system_prompt
            )

            # Crea la configuración (incluye historial de conversación)
            self.config = self.agent.set_config()

            # Genera mensaje de bienvenida desde el agente (en otro hilo)
            welcome_msg = await asyncio.to_thread(
                self.agent.chat_handler,
                "Preséntate con un mensaje de bienvenida personalizado para el usuario.",
                self.config
            )

            # Convierte el mensaje de texto a voz
            await self.generate_tts(welcome_msg)

            return {
                "message": "Datos del usuario recibidos.",
                "welcome_msg": welcome_msg
            }

        # Endpoint que recibe un archivo de audio, lo transcribe, genera respuesta y devuelve ambos
        @self.local_server.post("/receive")
        async def receive(file: UploadFile = File(...)):
            """
            Recibe audio de entrada del usuario, lo transcribe y genera una respuesta hablada.
            """
            # Genera nombre de archivo temporal único
            filename = f"{uuid4().hex}_{file.filename}"
            filepath = os.path.join(self.upload_folder, filename)

            try:
                # Guarda el archivo en disco
                async with aiofiles.open(filepath, 'wb') as out_file:
                    content = await file.read()
                    await out_file.write(content)

                # Transcribe el audio a texto
                result = await asyncio.to_thread(
                    self.transcriptor.transcribe,
                    audio=filepath,
                    language="spanish"
                )
                os.remove(filepath)
                transcription = result["text"]

                # Procesa la transcripción con el agente
                response = await asyncio.to_thread(
                    self.agent.chat_handler,
                    transcription,
                    self.config
                )

                # Convierte la respuesta en audio
                await self.generate_tts(response)

                return {"transcription": transcription, "response": response}

            except Exception as e:
                logging.error(traceback.format_exc())
                return JSONResponse(status_code=500, content={"error": str(e)})

        # Endpoint que devuelve el audio generado
        @self.local_server.get("/audio")
        def send_response_audio():
            """
            Devuelve el archivo de audio con la respuesta del asistente.
            """
            if os.path.exists(self.response_audio_path):
                return FileResponse(path=self.response_audio_path, media_type="audio/mpeg")
            else:
                return JSONResponse(status_code=404, content={"error": "No se encontró el audio."})

    async def generate_tts(self, text: str):
        """
        Convierte texto en audio usando el modelo TTS de Edge (voz en español).
        """
        tts = edge_tts.Communicate(text, voice="es-ES-XimenaNeural", rate="+10%")

        # Elimina el audio anterior si existe
        if os.path.exists(self.response_audio_path):
            os.remove(self.response_audio_path)

        # Guarda el nuevo archivo de audio
        await tts.save(self.response_audio_path)


# Lanza el servidor si se ejecuta como script principal
if __name__ == "__main__":
    import uvicorn
    from pyngrok import ngrok, conf

    # Token de autenticación para ngrok (reemplazar por el real)
    conf.get_default().auth_token = "<YOUR_NGROK_AUTH_TOKEN>"

    port = 5000
    public_url = ngrok.connect(port)
    print(f"Servidor alojado en: {public_url}")

    server = Backend()
    uvicorn.run(server.local_server, host="0.0.0.0", port=port)
