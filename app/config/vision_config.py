from __future__ import annotations


"""
Vision Configuration
Configures domain detection settings for vision analysis.
Model configuration is now handled by LLMConfig.vision (VisionLLMConfig).
"""


from pydantic import BaseModel, Field


class ImageGenerationConfig(BaseModel):
    """
    Configuration for image generation using OpenAI-compatible API (DALL-E 3, etc.).
    """

    enabled: bool = True
    api_key: str | None = None  # Loaded from .env (OPENAI_IMAGE_API_KEY)
    base_url: str | None = None  # Optional custom base URL for OpenAI-compatible APIs
    model: str = "dall-e-3"
    temperature: float = 1.0  # Only used for some compatible APIs


class ImageStorageConfig(BaseModel):
    """
    Configuration for image storage using imgbb API.
    Used to persist generated images.
    """

    enabled: bool = True
    api_key: str | None = None  # Loaded from .env (IMGBB_STORAGE_API_KEY)
    upload_url: str = "https://api.imgbb.com/1/upload"
    expiration: int | None = None  # Optional expiration in seconds (60-15552000), None = never


class VisionConfig(BaseModel):
    """
    Vision configuration for domain detection.
    Model configuration is now in LLMConfig.vision.
    """

    # Domain detection settings
    food_related_keywords: list[str] = Field(
        default_factory=lambda: [
            "菜品", "食材", "烹饪", "做菜", "食物", "美食", "饭菜",
            "炒", "煮", "蒸", "烤", "煎", "炸", "焖", "炖",
            "蔬菜", "水果", "肉类", "海鲜", "调料", "配料",
            "早餐", "午餐", "晚餐", "甜点", "饮品",
            "厨房", "刀工", "火候", "调味"
        ]
    )
