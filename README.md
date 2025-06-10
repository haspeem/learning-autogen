# learning-autogen

<!-- 项目徽章区 -->
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
![AutoGen Version](https://img.shields.io/badge/AutoGen-v0.2.7-green)

# 🚀 AutoGen 智能体开发生态实验室

<div align="center">
  <img src="docs/assets/autogen_workflow.png" alt="AutoGen架构图" width="600">
</div>

## 📜 项目愿景
构建可复现的AutoGen学习路径，涵盖从基础对话系统到企业级智能体应用的完整知识体系，包含：
- **基础层**：核心API用法/调试技巧
- **进阶层**：多智能体协作模式
- **应用层**：行业解决方案模板

## 📚 AutoGen官方教程路径

```mermaid
graph TB
    A[安装与环境配置] --> B[快速开始]
    B --> C[迁移指南 v0.2 → v0.4]
    C --> D[教程]
    
    D --> E[系统介绍]
    D --> F[模型使用]
    D --> G[消息管理]
    D --> H[智能体开发]
    D --> I[团队协作]
    D --> J[人机交互]
    J --> K[终止条件]
    J --> L[状态管理]
    
    style J fill:#c4ffc4,stroke:#00a000,stroke-width:2px
​​当前学习位置​​：已完成人机交互(Human-in-the-Loop)部分，正在开发智能体围棋(五子棋)博弈系统