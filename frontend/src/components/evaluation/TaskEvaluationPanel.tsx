/**
 * 任务级评测面板（P1）
 * 触发黄金任务集端到端评测，展示完成率与失败列表。
 */
import { useState } from 'react';
import { CheckCircle2, Loader2, Play, RefreshCcw, XCircle } from 'lucide-react';
import { createJsonHeaders, createAuthHeaders } from '../../services/api/client';
import { API_BASE } from '../../constants';

interface TaskEvalResult {
  datasets?: Record<string, DatasetResult>;
  job_id?: string;
  status?: string;
  output?: unknown;
}

interface DatasetResult {
  completion_rate: number;
  achieved: number;
  total_tasks: number;
  tasks?: Array<{
    task: string;
    achieved: boolean;
    reason?: string;
    judge?: string;
  }>;
  failures?: Array<{
    task: string;
    reason?: string;
  }>;
}

export function TaskEvaluationPanel({ token }: { token?: string }) {
  const [result, setResult] = useState<TaskEvalResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runEvaluation = async () => {
    if (!token) return;
    setRunning(true);
    setError(null);
    try {
      const response = await fetch(
        `${API_BASE}/evaluation/tasks/run?background=false`,
        {
          method: 'POST',
          headers: createJsonHeaders(token),
        }
      );
      if (!response.ok) {
        const msg = await response.text();
        throw new Error(msg || `HTTP error! status: ${response.status}`);
      }
      setResult(await response.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : '评测运行失败');
    } finally {
      setRunning(false);
    }
  };

  const checkJob = async (jobId: string) => {
    if (!token) return;
    setRunning(true);
    try {
      const response = await fetch(
        `${API_BASE}/evaluation/tasks/latest?job_id=${jobId}`,
        { headers: createAuthHeaders(token) }
      );
      if (!response.ok) throw new Error('查询失败');
      const job = await response.json();
      if (job.status === 'completed') {
        setResult({ datasets: (job.output as TaskEvalResult['datasets']) || {} });
        setRunning(false);
      } else if (job.status === 'failed') {
        setError(job.error || '评测失败');
        setRunning(false);
      } else {
        setTimeout(() => checkJob(jobId), 2000);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : '查询失败');
      setRunning(false);
    }
  };

  const datasets = result?.datasets || {};
  const entries = Object.entries(datasets);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
            任务级端到端评测
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            黄金任务集 → Agent 完整执行 → 判定任务是否达成（完成率）
          </p>
        </div>
        <button
          onClick={runEvaluation}
          disabled={running}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white text-sm font-medium"
        >
          {running ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} />}
          {running ? '评测中...' : '运行评测'}
        </button>
      </div>

      {error && (
        <div className="text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      {entries.length === 0 && !running && (
        <div className="text-center py-12 text-sm text-gray-400 border border-dashed border-gray-300 dark:border-gray-700 rounded-xl">
          点击"运行评测"执行黄金任务集，查看端到端任务完成率
        </div>
      )}

      {entries.map(([name, ds]) => (
        <div key={name} className="rounded-xl border border-gray-200 dark:border-gray-800 p-4 bg-white dark:bg-gray-900/50">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
              任务集: {name}
            </h3>
            <div className="flex items-center gap-3 text-sm">
              <span className="text-gray-500">
                {ds.achieved}/{ds.total_tasks} 达成
              </span>
              <span className={`font-semibold ${ds.completion_rate >= 0.8 ? 'text-emerald-500' : ds.completion_rate >= 0.5 ? 'text-amber-500' : 'text-red-500'}`}>
                {(ds.completion_rate * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {(ds.failures || []).length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-gray-500">失败用例:</p>
              {ds.failures!.map((f, i) => (
                <div key={i} className="flex items-start gap-2 text-sm">
                  <XCircle size={15} className="text-red-500 mt-0.5 shrink-0" />
                  <div>
                    <span className="text-gray-700 dark:text-gray-300">{f.task}</span>
                    {f.reason && (
                      <span className="text-xs text-gray-400 block">{f.reason}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {(ds.failures || []).length === 0 && ds.total_tasks > 0 && (
            <div className="flex items-center gap-2 text-sm text-emerald-600 dark:text-emerald-400">
              <CheckCircle2 size={15} /> 全部任务达成
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
