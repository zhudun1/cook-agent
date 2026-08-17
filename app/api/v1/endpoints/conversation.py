# app/api/v1/endpoints/conversation.py
"""
Conversation API endpoints for multi-turn chat with RAG integration.
Includes security features: input validation, prompt injection protection.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.services.conversation_service import conversation_service
from app.security.dependencies import check_message_security

logger = logging.getLogger(__name__)
router = APIRouter()

# Constants for input validation
MAX_MESSAGE_LENGTH = settings.MAX_MESSAGE_LENGTH  # 10000 characters
MAX_IMAGE_SIZE_MB = settings.MAX_IMAGE_SIZE_MB  # 5 MB
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


class ImageData(BaseModel):
    """Image data for multimodal requests."""
    data: str  # Base64 encoded image data
    mime_type: str = "image/jpeg"  # MIME type of the image

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        """Validate image MIME type."""
        if v not in ALLOWED_IMAGE_TYPES:
            raise ValueError(f"不支持的图片类型: {v}. 支持: {', '.join(ALLOWED_IMAGE_TYPES)}")
        return v

    @field_validator("data")
    @classmethod
    def validate_image_size(cls, v: str) -> str:
        """Validate base64 image size."""
        try:
            # Calculate approximate decoded size
            decoded_size = len(v) * 3 / 4
            if decoded_size > MAX_IMAGE_SIZE_BYTES:
                raise ValueError(f"图片大小超过限制 ({MAX_IMAGE_SIZE_MB}MB)")
        except Exception:
            pass  # If we can't calculate size, let it through for now
        return v


class ConversationRequest(BaseModel):
    """Request model for conversation endpoint."""
    message: str = Field(..., max_length=MAX_MESSAGE_LENGTH)
    conversation_id: Optional[str] = None
    stream: bool = True
    extra_options: Optional[Dict[str, Any]] = None  # e.g., {"web_search": true}
    images: Optional[List[ImageData]] = Field(
        default=None,
        description="List of images (base64 encoded) for multimodal understanding",
        max_length=5,  # Max 5 images per request
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate message content."""
        if not v or not v.strip():
            raise ValueError("消息不能为空")
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"消息长度超过限制 ({MAX_MESSAGE_LENGTH} 字符)")
        return v


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    conversation_id: str
    messages: list


class ConversationSummary(BaseModel):
    """Summary model for listing conversations."""
    id: str
    title: Optional[str] = None
    created_at: str
    updated_at: str
    message_count: int
    last_message_preview: str | None = None


@router.post("/conversation")
async def conversation(request: ConversationRequest, http_request: Request):
    """
    Handle a conversation message with optional RAG integration.

    The endpoint automatically detects whether the query needs knowledge
    base retrieval (RAG) or can be answered directly by the LLM.

    **Request Body:**
    - `message`: The user's input message
    - `conversation_id`: Optional ID for continuing a conversation
    - `stream`: Whether to stream the response (default: true)
    - `extra_options`: Optional features object, e.g., `{"web_search": true}`
    - `images`: Optional list of images for multimodal understanding

    **Response (SSE stream when stream=true):**
    ```
    data: {"type": "vision", "data": {"is_food_related": true, "intent": "...", "description": "..."}}
    data: {"type": "intent", "data": {"need_rag": true, "intent": "recipe_search", "reason": "..."}}
    data: {"type": "web_search", "data": {"confidence": 8, "reason": "...", "should_search": true}}
    data: {"type": "thinking", "content": "重写后的检索语句：番茄炒蛋的做法"}
    data: {"type": "text", "content": "..."}
    data: {"type": "sources", "data": [...]}
    data: {"type": "done", "conversation_id": "..."}
    ```
    """
    # ==========================================================================
    # Security Check: Use unified security check function
    # ==========================================================================
    secured_message = await check_message_security(request.message, http_request)

    logger.info(f"Received conversation request: '{secured_message[:50]}...', images={len(request.images) if request.images else 0}")
    
    # Convert images to service format
    images_data = None
    if request.images:
        images_data = [
            {"data": img.data, "mime_type": img.mime_type}
            for img in request.images
        ]
    
    # Use queue-based approach to ensure backend continues even if client disconnects
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    user_id = getattr(http_request.state, "user_id", None)

    async def process_in_background():
        """Background task that processes the chat and puts results in queue.

        This task runs independently from the client connection, ensuring
        messages are saved to database even if client refreshes/disconnects.
        """
        try:
            async for chunk in conversation_service.chat(
                message=secured_message,
                conversation_id=request.conversation_id,
                user_id=user_id,
                stream=True,
                extra_options=request.extra_options,
                images=images_data,
            ):
                await queue.put(chunk)
        except Exception as e:
            logger.error(f"Background processing error: {e}", exc_info=True)
        finally:
            await queue.put(None)  # Signal completion

    async def stream_from_queue() -> AsyncGenerator[str, None]:
        """Stream data from queue to client.

        If client disconnects, this generator stops but the background
        task continues processing to ensure messages are saved.
        """
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        except asyncio.CancelledError:
            # Client disconnected (e.g., page refresh)
            # Background task continues running independently
            logger.info("Stream cancelled by client, backend continues processing in background")
            # Don't raise - let the background task complete

    try:
        if request.stream:
            # Start background task BEFORE returning response
            # This ensures processing continues even if client disconnects
            asyncio.create_task(process_in_background())

            return StreamingResponse(
                stream_from_queue(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable nginx buffering
                }
            )
        else:
            # Non-streaming: collect all chunks
            full_response = ""
            sources = []
            conv_id = None
            intent_data = None
            
            async for event in conversation_service.chat(
                message=secured_message,
                conversation_id=request.conversation_id,
                user_id=getattr(http_request.state, "user_id", None),
                stream=False,
                extra_options=request.extra_options,
                images=images_data,
            ):
                # Parse SSE event
                if event.startswith("data: "):
                    import json
                    data = json.loads(event[6:].strip())
                    
                    if data["type"] == "text":
                        full_response += data["content"]
                    elif data["type"] == "sources":
                        sources = data["data"]
                    elif data["type"] == "done":
                        conv_id = data["conversation_id"]
                    elif data["type"] == "intent":
                        intent_data = data["data"]
                    elif data["type"] == "thinking":
                        # Thinking events are informational; no aggregation needed for non-streaming mode
                        continue
            
            return {
                "conversation_id": conv_id,
                "response": full_response,
                "sources": sources,
                "intent": intent_data
            }
            
    except Exception as e:
        logger.error(f"Error processing conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")


@router.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """
    Get the history of a conversation.
    
    **Parameters:**
    - `conversation_id`: The ID of the conversation
    
    **Response:**
    ```json
    {
        "conversation_id": "...",
        "messages": [
            {
                "role": "user",
                "content": "...",
                "timestamp": "...",
                "sources": null,
                "intent": null
            },
            {
                "role": "assistant",
                "content": "...",
                "timestamp": "...",
                "sources": [...],
                "intent": "recipe_search"
            }
        ]
    }
    ```
    """
    history = await conversation_service.get_conversation_history(conversation_id)
    
    if history is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=history
    )


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """
    Clear/delete a conversation.
    
    **Parameters:**
    - `conversation_id`: The ID of the conversation to delete
    """
    success = await conversation_service.clear_conversation(conversation_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"message": "Conversation cleared successfully"}


class UpdateTitleRequest(BaseModel):
    """Request model for updating conversation title."""
    title: str


@router.put("/conversation/{conversation_id}/title")
async def update_conversation_title(conversation_id: str, request: UpdateTitleRequest):
    """
    Update the title of a conversation.
    
    **Parameters:**
    - `conversation_id`: The ID of the conversation
    - `title`: The new title for the conversation
    """
    success = await conversation_service.update_conversation_title(conversation_id, request.title)
    
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"message": "Title updated successfully"}


class ConversationListResponse(BaseModel):
    """Response model for listing conversations."""
    conversations: list[ConversationSummary]
    total_count: int
    limit: int
    offset: int


@router.get("/conversation")
async def list_conversations(
    http_request: Request,
    limit: int = 50,
    offset: int = 0,
) -> ConversationListResponse:
    """List all conversations for the current user (PostgreSQL store).
    
    **Query Parameters:**
    - `limit`: Maximum number of conversations to return (default: 50)
    - `offset`: Number of conversations to skip (default: 0)
    """
    conversations, total_count = await conversation_service.list_conversations(
        user_id=getattr(http_request.state, "user_id", None),
        limit=limit,
        offset=offset,
    )
    return ConversationListResponse(
        conversations=[ConversationSummary(**c) for c in conversations],
        total_count=total_count,
        limit=limit,
        offset=offset,
    )
