# Taller 8 - LangChain LLM Chain Tutorial

## Descripción

Este repositorio contiene la implementación del tutorial básico de LangChain, enfocado en la construcción de cadenas LLM (Large Language Model) y la creación de un sistema RAG (Retrieval-Augmented Generation) básico. El proyecto demuestra cómo utilizar LangChain para interactuar con modelos de lenguaje, realizar búsquedas semánticas y construir agentes inteligentes que pueden recuperar y utilizar información contextual.

## Arquitectura y Componentes

### Componentes Principales

1. **LangChain Core**: Framework principal para orquestar las interacciones con LLMs
2. **OpenAI Integration**: Integración con los modelos GPT-4 y embeddings de OpenAI
3. **Vector Store (InMemory)**: Almacenamiento en memoria de vectores para búsqueda semántica
4. **Web Loader**: Herramienta para cargar y extraer contenido de páginas web
5. **Text Splitter**: Componente para dividir documentos largos en chunks manejables
6. **Agentes**: Sistemas autónomos que pueden utilizar herramientas para responder consultas

### Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      Usuario/Cliente                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangChain Agent                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         LLM (GPT-4) + System Prompt                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Tools (Herramientas)                      │  │
│  │    - retrieve_context: Búsqueda en Vector Store       │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              InMemory Vector Store                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │    Embeddings (OpenAI text-embedding-3-large)         │  │
│  │    - 63 documentos divididos                          │  │
│  │    - Búsqueda por similitud semántica                 │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Fuente de Datos                             │
│    Web Scraping: Blog Post sobre Agentes LLM                │
│    (https://lilianweng.github.io/posts/2023-06-23-agent/)   │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de Datos

1. **Carga de Datos**: Se extrae contenido de una página web usando `WebBaseLoader`
2. **Procesamiento**: El texto se divide en chunks usando `RecursiveCharacterTextSplitter`
3. **Vectorización**: Cada chunk se convierte en embeddings usando OpenAI
4. **Almacenamiento**: Los vectores se guardan en un `InMemoryVectorStore`
5. **Consulta**: El usuario hace una pregunta
6. **Recuperación**: El agente busca los chunks más relevantes
7. **Generación**: El LLM genera una respuesta usando el contexto recuperado

## Requisitos Previos

- Python 3.10 o superior
- Una cuenta de OpenAI con acceso a la API
- API Key de OpenAI
- (Opcional) API Key de LangSmith para trazabilidad

## Instalación de Dependencias

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/CamilaTorres08/taller8_AREP_langchain.git
cd taller8_AREP_langchain
```

### Paso 2: Crear un Entorno Virtual

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar las Dependencias

```bash
pip install langchain langchain-text-splitters langchain-community bs4
pip install langchain-openai
pip install langchain-core
pip install python-dotenv
pip install jupyter notebook
```

### Paso 4: Configurar Variables de Entorno

Cree una cuenta en Langsmith y obtenga la api key

<img width="1022" height="776" alt="image" src="https://github.com/user-attachments/assets/c2ac085e-4724-4f17-a832-7ed50f9c6e7a" />

Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```env
OPENAI_API_KEY=tu_api_key_aquí
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=tu_langsmith_key_aquí (opcional)
```


## Instrucciones de Ejecución

### Opción 1: Ejecutar con Jupyter Notebook

1. Inicia Jupyter Notebook:
```bash
jupyter notebook
```

2. Abre el archivo `taller8.ipynb` en tu navegador

3. Ejecuta las celdas secuencialmente presionando `Shift + Enter`

### Opción 2: Ejecutar desde Python

Si prefieres ejecutar el código como script de Python, puedes convertir el notebook:

```bash
jupyter nbconvert --to script taller8.ipynb
python taller8.py
```

## Descripción de las Secciones del Código

### 1. Instalación de Paquetes (Celdas 0, 2, 3, 5, 7)
Instala todas las dependencias necesarias de LangChain y sus integraciones.

### 2. Configuración de API Keys (Celdas 1, 4, 6)
Configura las variables de entorno para OpenAI y LangSmith.

```python
from dotenv import load_dotenv
import os

load_dotenv()              
api_key = os.getenv("OPENAI_API_KEY")
```

### 3. Inicialización de Embeddings (Celda 6)
Crea el modelo de embeddings que convertirá texto en vectores:

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### 4. Creación del Vector Store (Celda 8)
Inicializa el almacén de vectores en memoria:

```python
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)
```

### 5. Carga de Documentos Web (Celda 9)
Extrae contenido de una página web sobre agentes LLM:

```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
```

**Resultado**: Carga un documento con 43,047 caracteres.

### 6. División de Texto (Celda 11)
Divide el documento en chunks más pequeños:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
```

**Resultado**: 63 sub-documentos creados.

### 7. Almacenamiento en Vector Store (Celda 12)
Guarda los chunks vectorizados:

```python
document_ids = vector_store.add_documents(documents=all_splits)
```

### 8. Creación de Herramientas de Recuperación (Celda 13)
Define una herramienta que el agente puede usar para buscar información:

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
```

### 9. Creación del Agente (Celda 14)
Crea un agente que puede usar las herramientas:

```python
from langchain.agents import create_agent

tools = [retrieve_context]
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)
```

### 10. Consulta al Agente (Celda 15)
Hace una pregunta compleja que requiere múltiples búsquedas:

```python
query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
```

### 11. Agente con Middleware (Celdas 16-17)
Implementa un agente con inyección automática de contexto:

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)
    
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )
    
    return system_message

agent = create_agent(model, tools=[], middleware=[prompt_with_context])
```

## Ejemplos de Uso y Resultados

### Ejemplo 1: Consulta Simple

**Pregunta**: "What is task decomposition?"

**Respuesta del Agente**:
```
Task decomposition is the process of breaking down a complex task or goal into 
smaller, more manageable sub-tasks or steps. This allows an agent (such as an 
AI system or a human) to understand and plan how to achieve the overall objective 
by completing each part one at a time.

There are several ways to perform task decomposition:

1. Simple prompting with Large Language Models (LLMs):
   You can ask an LLM to list the steps for a specific task (e.g., "Steps for 
   planning a party:\n1."), or inquire about the subgoals needed to achieve a 
   goal (e.g., "What are the subgoals for writing a research paper?").

2. Task-specific instructions:
   For particular domains, using targeted prompts such as "Write a story outline" 
   can guide the model to generate a breakdown suitable for that scenario.

3. Human inputs:
   Humans can manually analyze and divide a big task into sub-tasks based on 
   their knowledge and experience.

4. LLM+P (LLM plus Planner):
   In this method, the LLM translates the problem into a formal language like 
   PDDL (Planning Domain Definition Language), invokes an external classical 
   planner to generate a step-by-step solution, and then translates the plan 
   back into natural language.
```

### Ejemplo 2: Consulta Compleja con Múltiples Búsquedas

**Pregunta**: "What is the standard method for Task Decomposition? Once you get the answer, look up common extensions of that method."

**Proceso del Agente**:
1. **Primera búsqueda**: Busca información sobre "standard method for Task Decomposition"
2. **Encuentra**: Chain of Thought (CoT) como método estándar
3. **Segunda búsqueda**: Busca extensiones del método
4. **Responde**: Con información sobre CoT y sus extensiones como Tree of Thoughts

### Ejemplo 3: Búsqueda Semántica en Vector Store

El sistema puede encontrar información relevante incluso si la pregunta no contiene las palabras exactas:

**Pregunta**: "How do AI agents plan their actions?"

**Resultado**: Recupera información sobre:
- Task Decomposition
- Planning components
- Chain of Thought
- Self-reflection mechanisms


## Características Técnicas

### Configuración de Text Splitter
- **Chunk Size**: 1000 caracteres
- **Chunk Overlap**: 200 caracteres
- **Tracking**: Índice de inicio en documento original

### Configuración de Búsqueda Semántica
- **Modelo de Embeddings**: `text-embedding-3-large` de OpenAI
- **Número de resultados** (k): 2 documentos más relevantes
- **Método**: Similitud coseno entre vectores

### Configuración del Agente
- **Modelo LLM**: GPT-4.1
- **Herramientas**: Búsqueda de contexto con formato de respuesta dual
- **Streaming**: Respuestas en tiempo real

## Conceptos Clave Implementados

1. **RAG (Retrieval-Augmented Generation)**:
   - Combina búsqueda de información con generación de texto
   - Mejora la precisión al proporcionar contexto relevante al LLM

2. **Vector Embeddings**:
   - Representación numérica de texto
   - Permite búsquedas semánticas (por significado, no por palabras exactas)

3. **Agentes LangChain**:
   - Sistemas autónomos que deciden qué herramientas usar
   - Pueden realizar múltiples pasos para responder consultas complejas

4. **Text Splitting**:
   - Divide documentos largos en chunks manejables
   - Mantiene coherencia con overlap entre chunks

5. **Tool Calling**:
   - El LLM decide cuándo y cómo usar herramientas
   - Formato estructurado para pasar información

## Referencias 

- **Tutorial Original**: [LangChain LLM Chain Tutorial](https://python.langchain.com/docs/tutorials/llm_chain/)
- **OpenAI API**: https://platform.openai.com/docs/

## Autor

**Andrea Camila Torres González**  

