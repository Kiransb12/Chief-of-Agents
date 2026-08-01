import os
from dotenv import load_dotenv

load_dotenv(override=True)  # Load environment variables

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Deepgram STT configuration
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-3")
DEEPGRAM_STT_URI = os.environ.get("DEEPGRAM_STT_URI", "wss://api.deepgram.com/v1/listen")
DEEPGRAM_ENDPOINTING = os.environ.get("DEEPGRAM_ENDPOINTING", "300")
DEEPGRAM_SMART_FORMAT = os.environ.get("DEEPGRAM_SMART_FORMAT", "true")

# Deepgram TTS configuration
DEEPGRAM_TTS_MODEL = os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-neptune-en")
DEEPGRAM_TTS_URI = os.environ.get("DEEPGRAM_TTS_URI", "https://api.deepgram.com/v1/speak")



# Cartesia TTS configuration
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY", "")
CARTESIA_MODEL_ID = os.environ.get("CARTESIA_MODEL_ID", "sonic-3.5")
CARTESIA_VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "5ee9feff-1265-424a-9d7f-8e4d431a12c7")
CARTESIA_VERSION = os.environ.get("CARTESIA_VERSION", "2026-03-01")
CARTESIA_TTS_URI = os.environ.get("CARTESIA_TTS_URI", "wss://api.cartesia.ai/tts/websocket")


# OpenAI Realtime configuration
OPENAI_REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
OPENAI_REALTIME_VOICE = os.environ.get("OPENAI_REALTIME_VOICE", "alloy")
OPENAI_REALTIME_URI = os.environ.get("OPENAI_REALTIME_URI", "wss://api.openai.com/v1/realtime")

# Gemini Live configuration
GEMINI_LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "models/gemini-2.5-flash-native-audio-latest")
GEMINI_LIVE_VOICE = os.environ.get("GEMINI_LIVE_VOICE", "Puck")
GEMINI_LIVE_URI = os.environ.get("GEMINI_LIVE_URI", "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")

# Priority order for voice providers
VOICE_PROVIDER_PRIORITY = ["deepgram_cartesia", "openai", "gemini"]



# Two-tier model strategy: a cheap/fast model triages every turn,
# a stronger model does the actual reasoning/answering.
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "llama-3.1-8b-instant")
REASONING_MODEL = os.environ.get("REASONING_MODEL", "llama-3.3-70b-versatile")

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")
COLLECTION_NAME = "personal_knowledge"

# Redis Configurations
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)
SESSION_STORE_BACKEND = os.environ.get("SESSION_STORE_BACKEND", "in-memory")

# Cost per million tokens in USD on Groq
MODEL_PRICING = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
}
