# 外部专业 Skills

此目录是桌面研究 Agent 的外部 Skill 安装位置。源码运行时使用项目根目录下的本目录；打包运行时使用 `PriceResearchAgent.exe` 同级的 `skills/` 目录。

每个可调用 Skill 使用独立子目录：

```text
skills/
└── professional-skill-name/
    ├── SKILL.md
    ├── references/   # 可选
    ├── assets/       # 可选
    └── scripts/      # 可选；第一阶段不会自动执行
```

应用自带两个核心 Skill（`price-exogenous-eda` 与 `price-forecastability-audit`），此目录用于追加你自己的专业流程。加载器读取 `SKILL.md` 的元数据和指令。Skill 可在 `metadata.protocol-file` 中声明包内相对路径的 YAML 领域研究协议；协议只能引用应用已注册函数，且必须完整覆盖该 Skill 的授权函数。加载器拒绝绝对路径和任何越出 Skill 目录的引用。外部 Skill 不能通过指令注册新函数，也不会自动执行 `scripts/` 中的代码。

如果需要从其他位置加载 Skill，可在 `.env` 中使用系统路径分隔符配置：

```dotenv
VPP_SKILL_PATHS=D:\research-skills;E:\shared-skills
```
