# GitHub 与 Zenodo 发布步骤

## 发布前必须完成

1. 再检查一次 `README.md`、`CITATION.cff`、作者姓名、单位和邮箱。
2. 仓库已采用 MIT License；第三方数据仍遵循 `DATASETS.md` 中列出的原始许可证。
3. 不要把 `external_data/` 中下载后的第三方原始数据提交到 Git；`.gitignore` 已默认排除这些文件。

## 使用 GitHub CLI 上传

在 PowerShell 中进入本文件夹，然后执行：

```powershell
cd "$HOME\Desktop\bounded-paired-degradation-fusion"
gh auth login
git add .
git commit -m "Initial reproducible release"
gh repo create bounded-paired-degradation-fusion --public --source . --remote origin --push
```

本地文件夹已经初始化为 `main` 分支，无需再次执行 `git init`。

成功后，仓库 URL 通常为：

```text
https://github.com/你的GitHub用户名/bounded-paired-degradation-fusion
```

本仓库包含一个约 61 MB 的结果文件，因此不要使用 GitHub 网页逐个上传；请使用 GitHub CLI、Git 命令行或 GitHub Desktop。

## 通过 Zenodo 获得 DOI

1. 登录 Zenodo，并连接 GitHub 账户。
2. 在 Zenodo 的 GitHub 页面点击同步，找到 `bounded-paired-degradation-fusion` 并启用。
3. 回到 GitHub，确认 README、CITATION、许可证和文件均正确。
4. 在 GitHub 创建 Release：标签 `v1.0.0`，标题 `Version 1.0.0`。
5. 等待 Zenodo 自动归档该 Release。
6. 打开 Zenodo 记录，复制版本 DOI；把 DOI 加入论文代码可用性声明、README 和后续版本的 `CITATION.cff`。

用于精确复现时，优先引用该 Release 的版本 DOI；概念 DOI 用于指向该软件的所有版本。

## 官方说明

- GitHub 本地代码上传：https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- Zenodo 启用 GitHub 仓库：https://help.zenodo.org/docs/github/enable-repository/
- Zenodo 归档 GitHub Release：https://help.zenodo.org/docs/github/archive-software/github-upload/
