# AGENTS.md — 会话约定

## GitHub 提交物格式（用户明确要求，优先级高）

提交到 GitHub 的所有文本 — commit message、PR 标题/描述、issue 评论、GitHub 上的 README/文档 — **不要按固定宽度（如 80/72 字符）自动换行**。
只在自然边界换行：句子、段落、列表项各占一行，行可以很长，由渲染端自动折行。
例外：代码块、表格、命令示例中的换行按其语法需要。

本仓库早期文档（docs/、STATUS.md 等）是固定宽度换行的历史产物，仅在未来"提交到 GitHub 的新文本"上执行本规则；既有上游 PR 草案（phase2/pr1/pr-draft.md）已按本规则重写。

## 其他既有约定

- Phase-1 证据文件（server.log、client.log、cache-before/after.txt、requests.jsonl、raw env/git 记录）语义内容永不改写。
- experiment_matrix.csv / docs/08 由 scripts/rebuild_experiment_matrix.py 生成，勿手改。
- 代理网络下 git 推拉用 scripts/git_retry.sh 重试；大对象走 codeload/gh api。
