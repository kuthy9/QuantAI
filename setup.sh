#!/bin/bash

# QuantAI AutoGen 项目设置脚本
echo "🚀 QuantAI AutoGen 项目设置开始..."

# 检查Python版本
echo "🔍 检查Python版本..."
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python版本检查通过: $(python3 --version)"
else
    echo "❌ Python版本过低，需要Python 3.9+，当前版本: $(python3 --version)"
    exit 1
fi

# 创建虚拟环境
echo "📦 创建虚拟环境..."
if [ -d "venv" ]; then
    echo "⚠️  虚拟环境已存在，是否重新创建? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "⬆️  升级pip..."
pip install --upgrade pip

# 安装核心依赖
echo "📚 安装核心依赖..."
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# 安装开发依赖
echo "🛠️  安装开发依赖..."
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements-dev.txt

# 验证安装
echo "✅ 验证安装..."
python -c "
import sys
print(f'Python版本: {sys.version}')

# 检查关键依赖
dependencies = [
    'autogen_core',
    'autogen_agentchat', 
    'openai',
    'anthropic',
    'fastapi',
    'pandas',
    'numpy',
    'pytest'
]

print('\n📦 依赖项检查:')
for dep in dependencies:
    try:
        __import__(dep)
        print(f'✅ {dep}: 已安装')
    except ImportError as e:
        print(f'❌ {dep}: 未安装 - {e}')
"

# 创建必要的目录
echo "📁 创建项目目录..."
mkdir -p logs
mkdir -p data
mkdir -p test_results
mkdir -p config

# 检查.env文件
echo "🔐 检查环境配置..."
if [ -f ".env" ]; then
    echo "✅ .env文件已存在"
else
    echo "⚠️  .env文件不存在，请确保已配置API密钥"
fi

# 设置Git hooks (如果是Git仓库)
if [ -d ".git" ]; then
    echo "🔗 设置Git hooks..."
    if [ -f "requirements-dev.txt" ]; then
        pip install pre-commit
        if [ -f ".pre-commit-config.yaml" ]; then
            pre-commit install
            echo "✅ Pre-commit hooks已安装"
        fi
    fi
fi

echo ""
echo "🎉 QuantAI AutoGen 项目设置完成！"
echo ""
echo "📋 下一步操作:"
echo "1. 确保 .env 文件包含所需的API密钥"
echo "2. 运行 './activate_env.sh' 激活环境"
echo "3. 运行 'python examples/complete_demo.py' 测试系统"
echo ""
echo "💡 提示:"
echo "- 使用 'source activate_env.sh' 激活环境"
echo "- 使用 'deactivate' 退出虚拟环境"
echo "- 查看 README.md 获取更多信息"
