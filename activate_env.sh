#!/bin/bash

# QuantAI AutoGen 虚拟环境激活脚本
echo "🚀 激活 QuantAI AutoGen 虚拟环境..."

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行 setup.sh"
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

# 验证环境
echo "✅ 虚拟环境已激活"
echo "📍 Python 版本: $(python --version)"
echo "📦 pip 版本: $(pip --version)"

# 显示关键依赖项
echo ""
echo "🔍 关键依赖项检查:"
python -c "
try:
    import autogen_core
    print('✅ autogen_core: 已安装')
except ImportError:
    print('❌ autogen_core: 未安装')

try:
    import autogen_agentchat
    print('✅ autogen_agentchat: 已安装')
except ImportError:
    print('❌ autogen_agentchat: 未安装')

try:
    import openai
    print('✅ openai: 已安装')
except ImportError:
    print('❌ openai: 未安装')

try:
    import anthropic
    print('✅ anthropic: 已安装')
except ImportError:
    print('❌ anthropic: 未安装')

try:
    import fastapi
    print('✅ fastapi: 已安装')
except ImportError:
    print('❌ fastapi: 未安装')

try:
    import pandas
    print('✅ pandas: 已安装')
except ImportError:
    print('❌ pandas: 未安装')

try:
    import pytest
    print('✅ pytest: 已安装')
except ImportError:
    print('❌ pytest: 未安装')
"

echo ""
echo "🎯 环境准备完成！您现在可以运行 QuantAI AutoGen 系统了。"
echo ""
echo "📚 常用命令:"
echo "  python examples/complete_demo.py          # 运行完整演示"
echo "  python examples/testing_integration_demo.py  # 运行测试演示"
echo "  python tests/test_runner.py               # 运行测试套件"
echo "  python -m quantai.api.server             # 启动API服务器"
echo ""
echo "💡 提示: 使用 'deactivate' 命令退出虚拟环境"
