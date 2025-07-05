#!/bin/bash
set -e

# QuantAI Production Startup Script
# This script handles production startup with proper initialization,
# configuration validation, and graceful error handling.

echo "🚀 Starting QuantAI Production System..."

# Environment validation
echo "📋 Validating environment configuration..."

# Required environment variables
REQUIRED_VARS=(
    "QUANTAI_ENV"
    "DATABASE_URL"
    "REDIS_URL"
)

# Check required variables
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

echo "✅ Environment validation passed"

# Wait for dependencies
echo "⏳ Waiting for dependencies..."

# Wait for PostgreSQL
echo "  - Waiting for PostgreSQL..."
until pg_isready -h postgres -p 5432 -U quantai; do
    echo "    PostgreSQL is unavailable - sleeping"
    sleep 2
done
echo "  ✅ PostgreSQL is ready"

# Wait for Redis
echo "  - Waiting for Redis..."
until redis-cli -h redis -p 6379 ping; do
    echo "    Redis is unavailable - sleeping"
    sleep 2
done
echo "  ✅ Redis is ready"

# Wait for ChromaDB
echo "  - Waiting for ChromaDB..."
until curl -f http://chroma:8000/api/v1/heartbeat; do
    echo "    ChromaDB is unavailable - sleeping"
    sleep 2
done
echo "  ✅ ChromaDB is ready"

# Database migrations (if needed)
echo "🔄 Running database migrations..."
python -c "
import asyncio
import asyncpg
import os

async def run_migrations():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        
        # Create tables if they don't exist
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS agent_status (
                agent_id VARCHAR(255) PRIMARY KEY,
                status VARCHAR(50) NOT NULL,
                last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            );
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                trade_id VARCHAR(255) PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                action VARCHAR(10) NOT NULL,
                quantity DECIMAL NOT NULL,
                price DECIMAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                agent_id VARCHAR(255),
                metadata JSONB
            );
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS risk_alerts (
                alert_id SERIAL PRIMARY KEY,
                alert_type VARCHAR(100) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE,
                metadata JSONB
            );
        ''')
        
        await conn.close()
        print('✅ Database migrations completed')
        
    except Exception as e:
        print(f'❌ Database migration failed: {e}')
        exit(1)

asyncio.run(run_migrations())
"

# Initialize logging directory
echo "📝 Setting up logging..."
mkdir -p /app/logs
touch /app/logs/quantai.log
touch /app/logs/agents.log
touch /app/logs/trades.log
touch /app/logs/risk.log

# Set proper permissions
chown -R quantai:quantai /app/logs
chmod 755 /app/logs
chmod 644 /app/logs/*.log

echo "✅ Logging setup complete"

# Configuration validation
echo "🔧 Validating system configuration..."
python -c "
import os
import json

# Validate trading mode
trading_mode = os.getenv('TRADING_MODE', 'paper')
if trading_mode not in ['paper', 'live']:
    print(f'❌ Invalid TRADING_MODE: {trading_mode}. Must be paper or live.')
    exit(1)

# Validate IBKR configuration
if trading_mode == 'live':
    required_ibkr_vars = ['IBKR_HOST', 'IBKR_PORT', 'IBKR_CLIENT_ID', 'IBKR_ACCOUNT']
    for var in required_ibkr_vars:
        if not os.getenv(var):
            print(f'❌ Required IBKR variable {var} not set for live trading')
            exit(1)

print('✅ Configuration validation passed')
print(f'📊 Trading Mode: {trading_mode}')
"

# System resource check
echo "💾 Checking system resources..."
python -c "
import psutil

# Check available memory
memory = psutil.virtual_memory()
if memory.available < 1024 * 1024 * 1024:  # 1GB
    print('⚠️  WARNING: Low memory available')
else:
    print(f'✅ Memory: {memory.available // (1024*1024)} MB available')

# Check disk space
disk = psutil.disk_usage('/')
if disk.free < 5 * 1024 * 1024 * 1024:  # 5GB
    print('⚠️  WARNING: Low disk space')
else:
    print(f'✅ Disk: {disk.free // (1024*1024*1024)} GB available')
"

# Start the application
echo "🎯 Starting QuantAI application..."

# Set up signal handlers for graceful shutdown
trap 'echo "🛑 Received shutdown signal, stopping QuantAI..."; kill -TERM $PID; wait $PID' TERM INT

# Start the main application
python -m quantai.main &
PID=$!

echo "✅ QuantAI started successfully (PID: $PID)"
echo "🌐 API server available at http://localhost:8000"
echo "📊 Dashboard available at http://localhost:8501"

# Wait for the process
wait $PID
