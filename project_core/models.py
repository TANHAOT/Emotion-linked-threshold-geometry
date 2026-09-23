MAIN_22 = [
    "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "claude-sonnet-4-5-20250929",
    "deepseek-v3", "deepseek-r1", "deepseek-v3.2", "gemini-2.0-flash", "gemini-2.5-pro", "gemini-3-pro-preview",
    "gpt-3.5-turbo-0125", "gpt-4.1-2025-04-14", "gpt-5-2025-08-07", "gpt-5.5", "o3-mini-2025-01-31", "o4-mini-2025-04-16",
    "qwen1.5-32b-chat", "qwen2.5-32b-instruct", "qwen3-32b", "qwen3-30b-a3b-instruct-2507", "qwen3-30b-a3b-thinking-2507",
    "qwen3-235b-a22b-instruct-2507", "qwen3-235b-a22b-thinking-2507",
]
CONTROL_4 = ["gpt-3.5-turbo-0125", "o3-mini-2025-01-31", "deepseek-v3", "deepseek-r1"]
TEMPERATURE_7 = ["deepseek-v3", "deepseek-r1", "gpt-4.1-2025-04-14", "claude-3-7-sonnet-20250219", "claude-sonnet-4-5-20250929", "gemini-2.5-pro", "gemini-3-pro-preview"]

TOKEN_LIMITS = {
    "gpt-3.5-turbo-0125": 4096,
    "o3-mini-2025-01-31": 99999,
    "o4-mini-2025-04-16": 99999,
}

def token_limit(model: str) -> int:
    if model in TOKEN_LIMITS:
        return TOKEN_LIMITS[model]
    if model.startswith("claude"):
        return 200000
    if model.startswith("deepseek"):
        return 16384
    if model.startswith("gemini"):
        return 60000
    return 32768
