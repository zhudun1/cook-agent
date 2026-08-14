import { ShieldAlert, Check, X } from 'lucide-react';
import type { PendingApproval } from '../../hooks/useAgent';

interface ApprovalCardProps {
  approval: PendingApproval;
  onDecide: (approvalId: string, approve: boolean) => void;
}

/**
 * HITL 审批卡片：Agent 请求调用危险工具时展示，
 * 用户批准后 Agent 继续执行，拒绝则 Agent 收到 APPROVAL_DENIED。
 */
export function ApprovalCard({ approval, onDecide }: ApprovalCardProps) {
  const argPreview = (() => {
    try {
      return JSON.stringify(approval.arguments, null, 2);
    } catch {
      return String(approval.arguments);
    }
  })();

  return (
    <div className="my-3 rounded-xl border border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20 p-4">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 text-amber-500">
          <ShieldAlert size={20} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-amber-800 dark:text-amber-200">
            需要你的审批：{approval.name}
          </div>
          <p className="text-xs text-amber-700 dark:text-amber-300 mt-1">
            Agent 请求调用该工具，请确认是否允许执行。
          </p>
          <pre className="mt-2 text-xs bg-white/60 dark:bg-black/20 rounded-lg p-2 overflow-x-auto text-gray-700 dark:text-gray-300 max-h-40">
            {argPreview}
          </pre>
          <div className="mt-3 flex gap-2">
            <button
              onClick={() => onDecide(approval.approval_id, true)}
              className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-medium transition-colors"
            >
              <Check size={14} /> 批准
            </button>
            <button
              onClick={() => onDecide(approval.approval_id, false)}
              className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-red-500 hover:bg-red-600 text-white text-xs font-medium transition-colors"
            >
              <X size={14} /> 拒绝
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
