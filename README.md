# Telegram AI Agent with RAG

AI-агент с Telegram-интерфейсом и **Retrieval-Augmented Generation (RAG)** для работы с внешней базой знаний.

Проект объединяет Telegram-бота с RAG-системой: пользователь отправляет запрос через Telegram, агент извлекает релевантную информацию из базы знаний и передаёт найденный контекст LLM для формирования ответа.

Такой подход позволяет использовать AI-агента не только для генерации ответов, но и для работы со специализированными документами и внутренними знаниями.

## Architecture

```text
                         ┌─────────────────────┐
                         │       User          │
                         │      Telegram       │
                         └──────────┬──────────┘
                                    │
                                    │ Message
                                    ▼
                         ┌─────────────────────┐
                         │   Telegram Bot      │
                         │                     │
                         │  Message Handler    │
                         └──────────┬──────────┘
                                    │
                                    │ User Query
                                    ▼
                         ┌─────────────────────┐
                         │    AI Agent         │
                         │                     │
                         │ Query Processing    │
                         │ Retrieval           │
                         │ Context Management  │
                         └──────────┬──────────┘
                                    │
                                    │ Search
                                    ▼
                         ┌─────────────────────┐
                         │    RAG Pipeline     │
                         │                     │
                         │ Embeddings          │
                         │ Retrieval           │
                         │ Ranking             │
                         │ Context Building    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Knowledge Base    │
                         │                     │
                         │ Documents            │
                         │ Chunks               │
                         │ Vector / Graph Data │
                         └──────────┬──────────┘
                                    │
                                    │ Relevant Context
                                    ▼
                         ┌─────────────────────┐
                         │        LLM          │
                         │                     │
                         │ Context + Query     │
                         │        ↓            │
                         │   Answer Generation │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Telegram       │
                         │      Response       │
                         └─────────────────────┘
```

## How It Works

Основной сценарий работы агента:

```text
User
  │
  │ "Найди информацию по вопросу..."
  ▼
Telegram Bot
  │
  ▼
AI Agent
  │
  ├── Analyze query
  │
  ├── Retrieve relevant knowledge
  │
  ├── Build context
  │
  └── Send context + query to LLM
  │
  ▼
Generated Answer
  │
  ▼
Telegram
```

### 1. User Query

Пользователь отправляет вопрос через Telegram.

### 2. Query Processing

Агент получает сообщение и подготавливает запрос для поиска в базе знаний.

### 3. Retrieval

RAG-компонент выполняет поиск релевантной информации в доступной базе знаний.

В зависимости от конфигурации могут использоваться различные подходы к retrieval:

* semantic search;
* vector search;
* hybrid search;
* graph-based retrieval;
* reranking.

### 4. Context Construction

Найденные документы или фрагменты отбираются и формируются в контекст для LLM.

```text
User Query
     +
Retrieved Context
     ↓
Prompt
     ↓
LLM
```

### 5. Answer Generation

LLM формирует ответ, используя не только собственные знания модели, но и найденную информацию из базы знаний.

### 6. Telegram Response

Сгенерированный ответ возвращается пользователю через Telegram.

---

# RAG Pipeline

RAG используется для подключения внешних знаний к AI-агенту.

```text
Documents
    ↓
Document Processing
    ↓
Chunking
    ↓
Embeddings
    ↓
Knowledge Base
    ↓
      ┌─────────────────┐
      │   User Query    │
      └────────┬────────┘
               ↓
          Retrieval
               ↓
      Relevant Chunks
               ↓
          Reranking
               ↓
      Context Assembly
               ↓
             LLM
               ↓
          Final Answer
```

## Почему RAG

Обычная LLM не имеет доступа к актуальной или внутренней информации проекта.

RAG решает эту проблему за счёт разделения:

**Knowledge → Retrieval → Generation**

Вместо передачи всей базы знаний модели система извлекает только релевантные фрагменты:

```text
Large Knowledge Base
        ↓
   Search / Retrieval
        ↓
Relevant Information
        ↓
        LLM
        ↓
     Answer
```

Это позволяет использовать агента для работы с:

* внутренней документацией;
* инструкциями;
* регламентами;
* специализированными знаниями;
* корпоративными документами;
* технической документацией;
* пользовательскими базами знаний.

---

# Agent + RAG

В проекте Telegram является интерфейсом взаимодействия, а RAG — механизмом доступа агента к внешним знаниям.

Это позволяет строить более сложный agentic workflow:

```text
                 User
                  │
                  ▼
             Telegram Bot
                  │
                  ▼
              AI Agent
                  │
          ┌───────┴────────┐
          │                │
          ▼                ▼
      RAG Search       LLM Reasoning
          │                │
          └───────┬────────┘
                  │
                  ▼
             Final Answer
                  │
                  ▼
               Telegram
```

В дальнейшем агент может самостоятельно определять, когда ему необходимо обратиться к базе знаний, а когда достаточно ответить напрямую.

---

# Retrieval Strategies

Архитектура может использовать несколько стратегий поиска.

### Semantic Search

Поиск информации по смысловой близости запроса и документов с использованием embeddings.

```text
Query
  ↓
Embedding
  ↓
Vector Search
  ↓
Top-K Documents
```

### Hybrid Search

Комбинация semantic/vector search и keyword-based поиска.

```text
             Query
               │
       ┌───────┴────────┐
       ▼                ▼
 Vector Search      Keyword Search
       │                │
       └───────┬────────┘
               ▼
          Result Fusion
               ↓
           Reranking
               ↓
         Relevant Context
```

Hybrid retrieval позволяет учитывать как семантическое сходство, так и точные совпадения терминов.

### Graph Retrieval

Для связанных сущностей знания могут представляться в виде графа:

```text
Entity A
   │
   ├── RELATED_TO ──→ Entity B
   │
   ├── PART_OF ─────→ Entity C
   │
   └── DESCRIBES ───→ Document
```

Графовый retrieval позволяет учитывать связи между сущностями и использовать их для формирования контекста.

---

# Project Structure

```text
tg-ai-agent/
│
├── chat_bot_tg/
│   ├── bot.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── ...
│
├── test-agentic-graph-rag-git/
│   └── Agentic Graph RAG system
│
└── README.md
```

### `chat_bot_tg`

Telegram-интерфейс AI-агента:

* получение сообщений;
* обработка пользовательских запросов;
* передача запросов AI-компоненту;
* возврат ответов пользователю.

### `test-agentic-graph-rag-git`

RAG-компонент проекта, предназначенный для работы с базой знаний и retrieval-процессом.

---

# Tech Stack

## AI / RAG

* **Python**
* **LLM**
* **Embeddings**
* **Retrieval-Augmented Generation**
* **Vector Search**
* **Semantic Search**
* **Hybrid Search**
* **Graph-based Retrieval**
* **Reranking**

## Telegram

* **Telegram Bot API**
* Telegram chatbot interface
* asynchronous message processing

## Infrastructure

* **Docker**
* **Docker Compose**
* environment-based configuration

## Knowledge Base

RAG-часть может быть расширена для работы с:

* vector databases;
* graph databases;
* document storage;
* embeddings;
* structured and unstructured data.

---

# Docker

Telegram-компонент проекта содержит `Dockerfile` и `docker-compose.yml`, поэтому приложение можно запускать в контейнере.

```bash
cd chat_bot_tg
docker compose up --build
```

Для запуска в background:

```bash
docker compose up -d --build
```

Остановка:

```bash
docker compose down
```

---

# Local Setup

## 1. Clone repository

```bash
git clone https://github.com/zitaisan/tg-ai-agent.git
cd tg-ai-agent
```

## 2. Install dependencies

```bash
cd chat_bot_tg
pip install -r requirements.txt
```

## 3. Configure environment

Создайте `.env` и добавьте необходимые credentials:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_api_key
```

Не добавляйте `.env` и API keys в Git.

## 4. Run Telegram Agent

```bash
python bot.py
```

---

# Example Workflow

Пользователь:

```text
Какие требования указаны в документации для запуска проекта?
```

Agent:

```text
1. Получает запрос через Telegram
2. Анализирует вопрос
3. Выполняет поиск по базе знаний
4. Находит релевантные документы
5. Формирует контекст
6. Передаёт контекст LLM
7. Генерирует ответ
8. Возвращает ответ в Telegram
```

Пример результата:

```text
В документации указаны следующие требования:

1. Python 3.x
2. Docker
3. Переменные окружения
4. Настроенная база данных

Источник: найденные документы из базы знаний.
```

---

# Advantages

### Контекстные ответы

Модель получает релевантную информацию непосредственно из базы знаний.

### Актуальность

Базу знаний можно обновлять без необходимости переобучать LLM.

### Масштабируемость

RAG-компонент можно подключать к различным источникам данных.

### Telegram Interface

Пользователю не требуется отдельное веб-приложение — взаимодействие происходит непосредственно через Telegram.

### Расширяемая Agent Architecture

В дальнейшем агент можно дополнить:

* tools;
* memory;
* external APIs;
* Graph RAG;
* multi-agent workflows;
* автоматическим выбором retrieval strategy.

---

# Future Improvements

Планируемые направления развития:

* полноценная conversational memory;
* query rewriting;
* автоматический выбор стратегии поиска;
* reranking retrieved documents;
* hybrid retrieval;
* Graph RAG;
* multi-step agentic retrieval;
* подключение нескольких источников знаний;
* metadata filtering;
* citation / source tracking;
* evaluation RAG pipeline;
* автоматическое тестирование качества retrieval;
* мониторинг LLM и retrieval pipeline;
* observability и tracing.

---

# Use Cases

Проект может использоваться как основа для AI-ассистентов:

* для внутренних корпоративных знаний;
* технической поддержки;
* работы с документацией;
* customer support;
* FAQ;
* обучения;
* поиска по большим коллекциям документов;
* автоматизации бизнес-процессов.

---

# Security

API keys и Telegram Bot Token должны храниться в переменных окружения.

```text
.env
```

не должен попадать в Git repository.

Для production-развёртывания рекомендуется:

* ограничить список пользователей Telegram;
* использовать отдельные credentials;
* ограничить доступ к базе знаний;
* не хранить secrets в исходном коде;
* использовать отдельного пользователя для запуска приложения.

---

# Development Roadmap

```text
Telegram Bot
     ↓
LLM Integration
     ↓
RAG Pipeline
     ↓
Hybrid Retrieval
     ↓
Reranking
     ↓
Graph RAG
     ↓
Agentic Retrieval
     ↓
Evaluation & Monitoring
```

---

# Author

**Taisia Zinchenko**

GitHub: [@zitaisan](https://github.com/zitaisan)

---

# Project Status

**AI / RAG prototype**

Проект демонстрирует интеграцию Telegram-интерфейса с LLM и RAG-подходом для построения AI-ассистента, способного отвечать на вопросы на основе внешней базы знаний.
