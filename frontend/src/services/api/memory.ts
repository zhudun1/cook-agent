/**
 * 长期记忆 API（P2）
 */
import { API_BASE, createAuthHeaders, createJsonHeaders, parseErrorResponse } from './client';

export interface MemoryItem {
  id: string;
  user_id: string;
  content: string;
  memory_type: 'preference' | 'goal' | 'restriction' | 'fact';
  importance: number;
  source: string;
  source_session_id?: string | null;
  access_count: number;
  created_at: string;
  last_accessed_at?: string | null;
}

export const MEMORY_TYPE_LABELS: Record<MemoryItem['memory_type'], string> = {
  preference: '偏好',
  goal: '目标',
  restriction: '限制',
  fact: '事实',
};

export async function listMemories(
  token?: string,
  memoryType?: MemoryItem['memory_type']
): Promise<{ memories: MemoryItem[]; total: number }> {
  const params = memoryType ? `?memory_type=${memoryType}` : '';
  const response = await fetch(`${API_BASE}/memory${params}`, {
    headers: createAuthHeaders(token),
  });
  if (!response.ok) {
    const msg = await parseErrorResponse(response);
    throw new Error(msg || `HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function addMemory(
  body: { content: string; memory_type: MemoryItem['memory_type']; importance?: number },
  token?: string
): Promise<{ message: string; memory?: MemoryItem }> {
  const response = await fetch(`${API_BASE}/memory`, {
    method: 'POST',
    headers: createJsonHeaders(token),
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const msg = await parseErrorResponse(response);
    throw new Error(msg || `HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function deleteMemory(
  memoryId: string,
  token?: string
): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE}/memory/${memoryId}`, {
    method: 'DELETE',
    headers: createAuthHeaders(token),
  });
  if (!response.ok) {
    const msg = await parseErrorResponse(response);
    throw new Error(msg || `HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function clearMemories(token?: string): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE}/memory`, {
    method: 'DELETE',
    headers: createAuthHeaders(token),
  });
  if (!response.ok) {
    const msg = await parseErrorResponse(response);
    throw new Error(msg || `HTTP error! status: ${response.status}`);
  }
  return response.json();
}
