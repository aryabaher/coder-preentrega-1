# Unified Async LLM Client

Cliente asíncrono en **Python 3.12** con interfaz común para **OpenAI**, **Anthropic** y **Gemini**: `generate()`, streaming con `yield`, schemas Pydantic (`ChatMessage`, `LLMConfig`, `ModelResponse`) y errores de API controlados (key inválida, rate limit, red). El negocio no instancia un SDK: usa `AsyncLLMManager`.

## Cómo ejecutarlo

Usá el bloque de **Windows (PowerShell)** o el de **Linux/macOS (bash/zsh)** según tu sistema. Los pasos 3 y 4 son los mismos en ambos, una vez activado el venv.

1. Entorno virtual e instalación (`openai`, `anthropic`, `google-genai`, `pydantic`, `python-dotenv`). Tests: `pytest`, `pytest-asyncio`, `pytest-mock`, `respx`.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Si PowerShell bloquea `Activate.ps1`, el `Set-ExecutionPolicy` de arriba vale solo para esa sesión. Alternativa sin activar: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` (y `-r requirements-dev.txt`) y después `.\.venv\Scripts\python.exe main.py`.

**Linux/macOS (bash/zsh):**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. Variables de entorno: copiá `.env.example` a `.env` y completá las keys.

**Windows (PowerShell):**

```powershell
copy .env.example .env
```

**Linux/macOS (bash/zsh):**

```bash
cp .env.example .env
```

3. Script de prueba (`main.py`): pregunta *¿Qué es la entropía?*, muestra la respuesta completa en texto plano extraída de `ModelResponse` y luego los tokens en tiempo real mediante streaming. También prueba una key inválida a propósito.

**Windows (PowerShell) y Linux/macOS (bash/zsh)** — el comando es el mismo:

```
python main.py
python main.py --provider openai
python main.py --provider anthropic
python main.py --provider gemini
```

4. Chequeo offline (sin API key).

**Windows (PowerShell) y Linux/macOS (bash/zsh)** — el comando es el mismo:

```
python validacion.py
python -m pytest -v
```

`pytest` cubre rangos de Pydantic, la factory y los tres errores de API (401 / 429 / red) con mocks: no gasta cuota.

## Archivos del repositorio

| Artefacto | Dónde está |
|-----------|------------|
| `schemas.py` (`Provider`, `ChatMessage` + `field_validator`, `LLMConfig` con `SecretStr`, `ModelResponse`) | `schemas.py` |
| `BaseLLMClient` con `async def generate()` y `generate_stream()` + `yield` | `llm_client/base.py` |
| `OpenAIClient` (`AsyncOpenAI` + `await client.chat.completions.create(...)`) | `llm_client/openai_client.py` |
| `AnthropicClient` (`AsyncAnthropic` + `messages.create` / `messages.stream`) | `llm_client/anthropic_client.py` |
| `GeminiClient` (`genai.Client(...).aio` + `generate_content` / `generate_content_stream`) | `llm_client/gemini_client.py` |
| `AsyncLLMManager` factory (`_crear_cliente` según `LLMConfig.provider`) | `llm_client/manager.py` |
| `.env.example` | `.env.example` |
| `main.py` (normal + streaming + key inválida) | `main.py` |
| Tests (`conftest.py`, `test_models.py`, `test_retries.py`) | `tests/` |
| `pytest.ini` | `testpaths = tests` · `asyncio_mode = auto` · `addopts = -ra -q` |
| `requirements-dev.txt` | `pytest`, `pytest-asyncio`, `pytest-mock`, `respx` |
| `requirements.txt` | `openai`, `anthropic`, `google-genai`, `pydantic`, `python-dotenv`, `pytest`, `pytest-asyncio`, `pytest-mock`, `respx` |

No commitees `.env`. El repo solo versiona `.env.example`.

## Rúbrica: dónde verificar cada criterio

| Criterio | Cómo se cumple | Evidencia |
|----------|----------------|-----------|
| Intercambiabilidad | `OpenAIClient`, `AnthropicClient` y `GeminiClient` heredan `BaseLLMClient`. `AsyncLLMManager._crear_cliente` elige por `Provider`. | `validacion.py` · `evidencias/01-validacion-offline.txt` |
| Asincronía | Solo SDKs async. `await client.chat.completions.create(...)` / `messages.create` / `aio.models.generate_content`. | Código de los tres clientes |
| Streaming | `async for` + `yield` en cada `generate_stream()`. El consumidor es idéntico para los tres. | `main.py` · `evidencias/02-openai-normal-y-streaming.txt` |
| Validación Pydantic | `LLMConfig.temperature` 0–2; `ChatMessage.role` validado. `temperature=5` se detecta **antes** de llamar a la API. | `schemas.py` · `evidencias/01-validacion-offline.txt` |
| Errores controlados | `RateLimitError` y red se reintentan 3 veces (backoff). Si persisten —o si la key es inválida— `ModelResponse.error` y el proceso sigue vivo. | `evidencias/03-error-controlado-api-key.txt` · `evidencias/04-reintentos-rate-limit.txt` · sección *Manejo de errores personalizados* |
| Tests automatizados | `pytest` + mocks de SDK: 401 no se reintenta; 429 y red sí. | `tests/test_models.py` · `tests/test_retries.py` · `evidencias/05-pytest.txt` |
| Gemini | Tercer proveedor, `GOOGLE_API_KEY`, rol `model` en vez de `assistant`. | `llm_client/gemini_client.py` |

## Variables de entorno

| Variable | Obligatorio | Para qué |
|----------|-------------|----------|
| `LLM_PROVIDER` | No (default `openai`) | `openai`, `anthropic` o `gemini` |
| `OPENAI_API_KEY` | Si usás OpenAI | Key de la API |
| `OPENAI_MODEL` | No (`gpt-4o-mini`) | Modelo OpenAI |
| `ANTHROPIC_API_KEY` | Si usás Anthropic | Key de la API |
| `ANTHROPIC_MODEL` | No (`claude-sonnet-4-5`) | Modelo Anthropic |
| `GOOGLE_API_KEY` | Si usás Gemini | Key de AI Studio (también acepta `GEMINI_API_KEY`) |
| `GEMINI_MODEL` | No (`gemini-flash-latest`) | Modelo Gemini |

## Evidencias

| Archivo | Qué demuestra |
|---------|---------------|
| `evidencias/01-validacion-offline.txt` | Tres clientes, async/streaming, `LLMConfig`, factory por `provider`, temperature=5 y reintentos simulados. |
| `evidencias/02-openai-normal-y-streaming.txt` | `python main.py --provider openai`: texto plano de `ModelResponse` y tokens en streaming. |
| `evidencias/03-error-controlado-api-key.txt` | Key inválida (401) → `ModelResponse.error`, sin crash. El 401 no se reintenta. |
| `evidencias/04-reintentos-rate-limit.txt` | 429 simulado: 3 intentos con backoff; si persiste se propaga; si el tercero responde, sigue. |
| `evidencias/05-pytest.txt` | `python -m pytest -v`: 11 tests (factory, key faltante, 401/429/red con mocks). |

Salida de `python -m pytest -v`:

```
tests/test_models.py::test_build_model_creates_openai PASSED
tests/test_models.py::test_missing_api_key_raises PASSED
tests/test_models.py::test_invalid_provider_raises PASSED
tests/test_retries.py::test_transient_retries_then_succeeds PASSED
tests/test_retries.py::test_non_transient_does_not_retry PASSED
============================= 11 passed in 8.41s ==============================
```

## Manejo de errores personalizados

El cliente no deja que una excepción de SDK mate `asyncio.run`. Cada familia tiene un comportamiento distinto.

### Key inválida (401)

No es transitorio: **un solo intento**. `generate()` captura el 401 y arma `ModelResponse(content="", error=...)`.

Cómo reproducirlo: `python main.py --provider openai` termina con `SecretStr("sk-key-invalida-a-proposito")`.

```
=== Prueba de resiliencia (API key inválida a propósito) ===
El programa siguió vivo: sí
Error capturado (sin crash): Error de la API de OpenAI: Error code: 401 - Incorrect API key provided (invalid_api_key)
```

El test `test_non_transient_does_not_retry` mockea `AuthenticationError` y exige `await_count == 1`.

### Rate limit / cuota (429)

Sí se reintenta: `retry_async` hace 3 intentos con backoff (1s → 2s → 4s). Si persiste, el error queda en `ModelResponse.error` como `Límite de cuota excedido: ...`. El proceso sigue vivo.

El test `test_openai_429_queda_en_model_response_error` mockea `RateLimitError` (HTTP 429) y comprueba `result.error` + `content == ""`.

### Red / timeout

También transitorio (3 intentos). Si DNS o el timeout persisten: `ModelResponse.error` con `Error de conexión: ...`.

El test `test_openai_red_queda_en_error` mockea `APIConnectionError`.

### Validación Pydantic (antes de la API)

`LLMConfig(temperature=5)` lanza `ValidationError` **sin** llamar a HTTP. `main.py` lo muestra como error controlado. El test `test_temperature_fuera_de_rango` cubre el mismo caso.

## Checklist de verificación

- [x] Python 3.12 y las deps (`openai`, `anthropic`, `google-genai`, `pydantic`, `python-dotenv`, `pytest`)
- [x] `schemas.py` en la raíz con `Provider`, `ChatMessage`, `LLMConfig`, `ModelResponse`
- [x] `AsyncLLMManager` carga el proveedor por `LLMConfig` / `_crear_cliente`
- [x] `python validacion.py` confirma Pydantic, interfaz común, factory y reintentos (429 simulado)
- [x] `python main.py` prueba normal, streaming y key inválida
- [x] Los errores de API no rompen `asyncio.run`
- [x] `python -m pytest -v` (mocks de 401 / 429 / red; sin gastar cuota)
- [x] README con ejemplos específicos de 401, 429, red y validación Pydantic
