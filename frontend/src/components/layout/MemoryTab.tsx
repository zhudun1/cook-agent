/**
 * 长期记忆面板（P2）
 * 查看 / 手动添加 / 删除 / 清空用户长期记忆。
 */
import { useCallback, useEffect, useState } from 'react';
import { Brain, Loader2, Plus, RefreshCcw, Trash2, X } from 'lucide-react';
import {
  listMemories,
  addMemory,
  deleteMemory,
  clearMemories,
  MEMORY_TYPE_LABELS,
} from '../../services/api/memory';
import type { MemoryItem } from '../../services/api/memory';

const TYPE_COLORS: Record<MemoryItem['memory_type'], string> = {
  preference: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
  goal: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  restriction: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
  fact: 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300',
};

export function MemoryTab({ token }: { token?: string }) {
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [content, setContent] = useState('');
  const [memoryType, setMemoryType] = useState<MemoryItem['memory_type']>('preference');

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listMemories(token);
      setMemories(data.memories);
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败');
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    load();
  }, [load]);

  const handleAdd = async () => {
    if (!content.trim()) return;
    try {
      await addMemory(
        { content: content.trim(), memory_type: memoryType, importance: 0.7 },
        token
      );
      setContent('');
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : '添加失败');
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteMemory(id, token);
      setMemories(prev => prev.filter(m => m.id !== id));
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败');
    }
  };

  const handleClear = async () => {
    if (!window.confirm('确定清空全部长期记忆吗？')) return;
    try {
      await clearMemories(token);
      setMemories([]);
    } catch (e) {
      setError(e instanceof Error ? e.message : '清空失败');
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-sm font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
            <Brain size={16} className="text-violet-500" />
            长期记忆（跨会话）
          </h4>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            自动从对话中沉淀你的偏好/目标/限制，也可手动维护
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={load}
            className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
            title="刷新"
          >
            <RefreshCcw size={15} />
          </button>
          {memories.length > 0 && (
            <button
              onClick={handleClear}
              className="p-2 rounded-lg text-red-500 hover:bg-red-50 dark:hover:bg-red-900/30"
              title="清空全部"
            >
              <Trash2 size={15} />
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-3 text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      {/* 添加表单 */}
      <div className="mb-4 space-y-2">
        <div className="flex gap-2">
          <input
            value={content}
            onChange={e => setContent(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleAdd()}
            placeholder="例如：我不吃香菜 / 目标是每周减重 0.5kg..."
            className="flex-1 px-3 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-violet-500"
          />
          <button
            onClick={handleAdd}
            disabled={!content.trim()}
            className="px-3 py-2 rounded-lg bg-violet-600 hover:bg-violet-700 disabled:opacity-40 text-white text-sm flex items-center gap-1"
          >
            <Plus size={14} /> 添加
          </button>
        </div>
        <div className="flex gap-1.5">
          {(Object.keys(MEMORY_TYPE_LABELS) as MemoryItem['memory_type'][]).map(t => (
            <button
              key={t}
              onClick={() => setMemoryType(t)}
              className={`px-2.5 py-1 rounded-full text-xs transition-colors ${
                memoryType === t
                  ? TYPE_COLORS[t]
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400'
              }`}
            >
              {MEMORY_TYPE_LABELS[t]}
            </button>
          ))}
        </div>
      </div>

      {/* 列表 */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {loading && memories.length === 0 && (
          <div className="flex justify-center py-8 text-gray-400">
            <Loader2 className="animate-spin" size={20} />
          </div>
        )}
        {!loading && memories.length === 0 && (
          <div className="text-center py-10 text-sm text-gray-400">
            暂无长期记忆。多和 Agent 聊聊你的饮食偏好与目标，它会自动为你记录。
          </div>
        )}
        {memories.map(m => (
          <div
            key={m.id}
            className="flex items-start gap-2 p-3 rounded-xl border border-gray-200/70 dark:border-gray-700/70 bg-white dark:bg-gray-900/50"
          >
            <span className={`shrink-0 mt-0.5 px-2 py-0.5 rounded-full text-[10px] font-medium ${TYPE_COLORS[m.memory_type]}`}>
              {MEMORY_TYPE_LABELS[m.memory_type]}
            </span>
            <span className="flex-1 text-sm text-gray-700 dark:text-gray-300 break-words">
              {m.content}
            </span>
            <button
              onClick={() => handleDelete(m.id)}
              className="shrink-0 p-1 rounded text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/30"
              title="删除"
            >
              <X size={14} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
