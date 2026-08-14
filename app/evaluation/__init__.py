# app/evaluation/__init__.py
"""
RAG Grounding-Truth 评测模块

与 app/services/evaluation_service.py（在线实时评估）互补：
- evaluation_service: 生产环境在线采样评估（faithfulness / answer_relevancy）
- 本模块: 基于 grounding truth 测试集的离线批量评测
  （context_precision / context_recall / answer_correctness），
  用于回归测试、模型对比与 RAG 系统上线前的完整验收

组件:
- dataset: 测试集加载与校验（JSONL）
- runner: 批量评测执行器（RAGAS + reference 指标）
- report: 评测报告生成（JSON + Markdown）
"""
