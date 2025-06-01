from typing import Annotated
from typing_extensions import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from llm_api import LLMApi
from langfuse.callback import CallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, ToolMessage
import os
import json_repair
from collections import deque

# Configuración de claves para Langfuse (monitorización de interacciones)
os.environ["LANGFUSE_PUBLIC_KEY"] = "<YOUR_LANGFUSE_PUBLIC_KEY>"
os.environ["LANGFUSE_SECRET_KEY"] = "<YOUR_LANGFUSE_SECRET_KEY>"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"  # Hosting europeo

# Estructura de estado del grafo conversacional
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # Historial de mensajes del modelo
    chat: List[Dict[str, Any]]  # Historial de la conversación en formato simple


class Agent:
    def __init__(self, llm: LLMApi, tools: list, system_prompt: str):
        self.graph_builder = StateGraph(State)
        self.llm = llm
        self.tools = tools
        self.system_prompt = {
            "role": "system",
            "content": system_prompt
        }
        self.state = {
            "messages": [],
            "chat": []
        }
        self.start = False

        # Nodo "manager": decide si se necesita usar herramientas externas
        def _manager(state: State):
            custom_response_format = {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query_type": {
                                "type": "string",
                                "description": "Tipo de consulta: 'search' o 'response'."
                            },
                            "justification": {
                                "type": "string",
                                "description": "Explicación del tipo de consulta."
                            },
                            "adapted_query": {
                                "type": "string",
                                "description": "Consulta modificada para búsqueda externa."
                            }
                        },
                        "required": ["query_type", "justification", "adapted_query"],
                        "additionalProperties": False
                    },
                    "name": "customized_output",
                    "description": "Clasificación de consulta del usuario",
                    "strict": True,
                }
            }

            # Prompt que define el criterio de clasificación de consultas
            question_type_prompt = [{
                "role": "system",
                "content": (
                    "Tu tarea es determinar si, en un momento específico de una conversación con un usuario, "
                    "es necesario aplicar una recuperación de información externa para continuar la conversación correctamente.\n\n"
                    "Clasifica la conversación únicamente en una de estas dos categorías:\n\n"
                    "1. 'search'  \n"
                    "Selecciona esta categoría si se necesita acceder a información externa (como internet o una base de datos) "
                    "para responder adecuadamente.  \n"
                    "Ejemplos:  \n"
                    "- \"¿Quién es Joaquín Sabina?\"  \n"
                    "- \"Dame una leyenda del Lagarto de la Malena.\"\n\n"
                    "2. 'response'  \n"
                    "Selecciona esta categoría si se puede responder directamente sin acceder a información externa.  \n"
                    "Ejemplos:  \n"
                    "- \"Hoy hace buen tiempo.\"  \n"
                    "- \"¿Cómo te llamas?\"\n\n"
                    "No clasifiques según si tú conoces la respuesta. Evalúa únicamente si es necesario recuperar información externa para continuar.\n\n"
                    "Incluye en tu respuesta:\n"
                    "- La clasificación ('search' o 'response').  \n"
                    "- Una justificación.  \n"
                    "- Una consulta adaptada (si aplica)."
                )
            }]

            # Construcción del contexto de clasificación usando el historial
            custom_chat = question_type_prompt + state.get("chat")[1:]
            response = json_repair.loads(
                self.llm.invoke(chat=custom_chat, response_format=custom_response_format)
            )

            state["messages"].append(AIMessage(content=str(response)))

            # Si requiere búsqueda, se invoca la herramienta
            if response.get("query_type") == 'search':
                state["messages"].append(AIMessage(content="", tool_calls=[{
                    "name": "retrieval_augmented_generation",
                    "args": {"query": response.get("adapted_query") or state["chat"][-1]['content']},
                    "id": "tool_call_id",
                    "type": "tool_call"
                }]))

            return state

        # Decide la siguiente ruta según si se ha invocado una herramienta
        def _route_tools(state: State):
            if messages := state.get("messages"):
                ai_message = messages[-1]
                if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                    return "tools"
            return "chatbot"

        # Nodo chatbot: genera la respuesta final
        def _chatbot(state: State):
            # Si el mensaje anterior era de herramienta, incorporar contexto
            if isinstance(state["messages"][-1], ToolMessage):
                context_msg = {
                    "role": "system",
                    "content": (
                        state["chat"][0]["content"] +
                        " Para continuar la conversación, quizás te sea útil el siguiente contexto: " +
                        state["messages"][-1].content +
                        ". Si esta información no es útil para continuar la conversación, tu máxima prioridad es "
                        "responder educadamente que no tienes suficiente información y proponer otro tema de conversación."
                    )
                }
                chat = [context_msg] + state["chat"][1:]
                response = self.llm.invoke(chat=chat)
                state["messages"] = state["messages"][:-1]
            else:
                response = self.llm.invoke(chat=state.get("chat", []))

            state["messages"].append(AIMessage(content=response))
            state["chat"].append({"role": "assistant", "content": response})
            return state

        # Construcción del grafo
        self.graph_builder.add_node("manager", _manager)
        self.graph_builder.add_node("tools", ToolNode(tools=self.tools))
        self.graph_builder.add_node("chatbot", _chatbot)

        self.graph_builder.add_conditional_edges("manager", _route_tools, {
            "tools": "tools",
            "chatbot": "chatbot"
        })

        self.graph_builder.add_edge(START, "manager")
        self.graph_builder.add_edge("tools", "chatbot")
        self.graph_builder.add_edge("chatbot", END)

        self.graph = self.graph_builder.compile()

    def invoke(self, user_message):
        """
        Ejecuta el grafo y devuelve solo la respuesta final.
        """
        config = {"configurable": {"thread_id": "1"}, "callbacks": [CallbackHandler()]}
        inputs = {
            "messages": HumanMessage(content=user_message),
            "chat": [self.system_prompt, {"role": "user", "content": user_message}]
        }
        stream = self.graph.stream(input=inputs, config=config, stream_mode="values")
        return deque(stream, maxlen=1)[-1]["chat"][-1]["content"]

    def chat(self):
        """
        Modo de conversación en bucle (útil para pruebas por consola).
        """
        config = {"configurable": {"thread_id": "1"}, "callbacks": [CallbackHandler()]}
        self.state["chat"].append(self.system_prompt)

        def _stream_graph_updates(user_message: str):
            self.state["chat"].append({"role": "user", "content": user_message})
            self.state["messages"].append(HumanMessage(content=user_message))
            response = self.graph.invoke(input=self.state, config=config, stream_mode="values")
            self.state["messages"] = response["messages"]
            self.state["chat"] = response["chat"]
            print(f"MARÍA >> {response['chat'][-1]['content']}")

        print("<-- INICIO DE EJECUCIÓN DEL AGENTE -->")
        while True:
            if not self.start:
                _stream_graph_updates("Preséntate con un mensaje de bienvenida personalizado para el usuario.")
                self.start = True
            else:
                try:
                    user_input = input("USUARIO >> ")
                    if user_input.lower() in ["q"]:
                        print("<-- FIN DE EJECUCIÓN DEL AGENTE -->")
                        break
                    _stream_graph_updates(user_input)
                except:
                    print("<-- ERROR DE EJECUCIÓN DEL AGENTE -->")
                    break

    def set_config(self):
        """
        Devuelve la configuración para Langfuse, útil para integración con el grafo.
        """
        config = {"configurable": {"thread_id": "1"}, "callbacks": [CallbackHandler()]}
        self.state["chat"].append(self.system_prompt)
        return config

    def chat_handler(self, user_message, config):
        """
        Maneja una interacción del usuario, actualiza el estado y devuelve la respuesta.
        """
        self.state["chat"].append({"role": "user", "content": user_message})
        self.state["messages"].append(HumanMessage(content=user_message))
        response = self.graph.invoke(input=self.state, config=config, stream_mode="values")
        self.state["messages"] = response["messages"]
        self.state["chat"] = response["chat"]
        return response["chat"][-1]["content"]
