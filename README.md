# Cliente LLM unificado y asíncrono

Interfaz común en Python 3.12 para llamar a **OpenAI** o **Anthropic** sin bloquear el event loop: generación completa, streaming de tokens y errores de API controlados (key inválida, rate limit, red).

## Quick path

1. Creá el entorno e instalá dependencias:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copiá las variables de entorno y completá las keys:

```powershell
copy .env.example .env
```

3. Corré la prueba (pregunta: *¿Qué es la entropía?* en modo normal y streaming):

```powershell
python main.py
python main.py --provider anthropic
```

Deberías ver el proveedor activo, la respuesta completa y luego los mismos tokens llegando de a poco.

## Variables de entorno

| Variable | Obligatorio | Para qué |
|----------|-------------|----------|
| `LLM_PROVIDER` | No (default `openai`) | `openai` o `anthropic` |
| `OPENAI_API_KEY` | Si usás OpenAI | Key de la API |
| `OPENAI_MODEL` | No (`gpt-4o-mini`) | Modelo OpenAI |
| `ANTHROPIC_API_KEY` | Si usás Anthropic | Key de la API |
| `ANTHROPIC_MODEL` | No (`claude-sonnet-4-5`) | Modelo Anthropic |
| `LLM_TIMEOUT` | No (`30`) | Timeout HTTP en segundos |

No commitees el archivo `.env`. El repo solo versiona `.env.example`.

## Qué hay en el repo

| Archivo | Rol |
|---------|-----|
| `llm_client/schemas.py` | Pydantic: `ChatMessage`, `ModelConfig` (temperatura 0–2, `max_tokens`), `ModelResponse` |
| `llm_client/base.py` | `BaseLLMClient` con `generate()` / `generate_stream()` y reintentos |
| `llm_client/openai_client.py` | `AsyncOpenAI` + `await client.chat.completions.create(...)` |
| `llm_client/anthropic_client.py` | `AsyncAnthropic` + `messages.create` / `messages.stream` |
| `llm_client/manager.py` | `AsyncLLMManager` elige el proveedor según `LLM_PROVIDER` |
| `main.py` | Script de validación (normal + streaming) |

`AsyncLLMManager` y `create_client("openai"|"anthropic")` instancian el mismo contrato, así el resto del código no se ata a un SDK.

## Comportamiento

- **Asíncrono:** solo clientes `AsyncOpenAI` / `AsyncAnthropic`. Nada de llamadas síncronas dentro de `async def`.
- **Streaming:** `async for` sobre el stream del SDK y `yield` de cada fragmento de texto.
- **Errores:** rate limit y fallos de red se reintentan (backoff exponencial, 3 intentos). Si persisten —o si la API key es inválida— `generate()` devuelve `ModelResponse.error` y el streaming levanta `LLMClientError`. El `asyncio.run` de `main.py` no se cae.
- **Temperatura:** `ModelConfig` la valida entre 0 y 2 (rango OpenAI). El SDK de Anthropic 1.3 ya no acepta `temperature` en `messages.create`; Claude usa el default del modelo.

## Checklist

- [ ] El venv es Python 3.12 (`python --version`)
- [ ] `.env` tiene la key del proveedor que vas a usar
- [ ] `python main.py` imprime la respuesta completa
- [ ] La sección *Modo streaming* muestra el texto de a poco, no de un solo golpe
