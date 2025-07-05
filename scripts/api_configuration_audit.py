#!/usr/bin/env python3
"""
API Configuration Audit Script
Tests connectivity and configuration for all API services in the QuantAI system.
"""

import os
import sys
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class APIConfigurationAuditor:
    """Comprehensive API configuration auditor for QuantAI system."""
    
    def __init__(self):
        self.results = {}
        self.load_environment()
    
    def load_environment(self):
        """Load environment variables from .env file."""
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
    
    async def test_openai_api(self) -> Dict[str, Any]:
        """Test OpenAI API connectivity and configuration."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            return {
                'status': 'FAILED',
                'error': 'API key not configured or using placeholder',
                'configured': False
            }
        
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.openai.com/v1/models', headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        model_count = len(data.get('data', []))
                        return {
                            'status': 'SUCCESS',
                            'configured': True,
                            'models_available': model_count,
                            'response_time_ms': response.headers.get('x-response-time', 'N/A')
                        }
                    else:
                        return {
                            'status': 'FAILED',
                            'error': f'HTTP {response.status}: {await response.text()}',
                            'configured': True
                        }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'configured': True
            }
    
    async def test_anthropic_api(self) -> Dict[str, Any]:
        """Test Anthropic API connectivity and configuration."""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key or api_key == 'your_anthropic_api_key_here':
            return {
                'status': 'FAILED',
                'error': 'API key not configured or using placeholder',
                'configured': False
            }
        
        try:
            headers = {
                'x-api-key': api_key,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
            
            # Test with a minimal message
            payload = {
                'model': 'claude-3-haiku-20240307',
                'max_tokens': 10,
                'messages': [{'role': 'user', 'content': 'Hello'}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post('https://api.anthropic.com/v1/messages', 
                                      headers=headers, json=payload, timeout=15) as response:
                    if response.status == 200:
                        return {
                            'status': 'SUCCESS',
                            'configured': True,
                            'model_tested': 'claude-3-haiku-20240307'
                        }
                    else:
                        return {
                            'status': 'FAILED',
                            'error': f'HTTP {response.status}: {await response.text()}',
                            'configured': True
                        }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'configured': True
            }
    
    async def test_polygon_api(self) -> Dict[str, Any]:
        """Test Polygon.io API connectivity and configuration."""
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key or api_key == 'your_polygon_api_key_here':
            return {
                'status': 'FAILED',
                'error': 'API key not configured or using placeholder',
                'configured': False
            }
        
        try:
            url = f'https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=1&apikey={api_key}'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': 'SUCCESS',
                            'configured': True,
                            'results_count': data.get('count', 0),
                            'status_code': data.get('status', 'unknown')
                        }
                    else:
                        return {
                            'status': 'FAILED',
                            'error': f'HTTP {response.status}: {await response.text()}',
                            'configured': True
                        }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'configured': True
            }
    
    async def test_alpaca_api(self) -> Dict[str, Any]:
        """Test Alpaca API connectivity and configuration."""
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key or api_key == 'your_alpaca_api_key_here':
            return {
                'status': 'FAILED',
                'error': 'API keys not configured or using placeholder',
                'configured': False
            }
        
        try:
            headers = {
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': secret_key
            }
            
            # Use paper trading endpoint
            base_url = 'https://paper-api.alpaca.markets'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{base_url}/v2/account', headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': 'SUCCESS',
                            'configured': True,
                            'account_status': data.get('status', 'unknown'),
                            'trading_blocked': data.get('trading_blocked', False),
                            'buying_power': float(data.get('buying_power', 0))
                        }
                    else:
                        return {
                            'status': 'FAILED',
                            'error': f'HTTP {response.status}: {await response.text()}',
                            'configured': True
                        }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'configured': True
            }
    
    async def test_alpha_vantage_api(self) -> Dict[str, Any]:
        """Test Alpha Vantage API connectivity and configuration."""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key or api_key == 'your_alpha_vantage_api_key_here':
            return {
                'status': 'FAILED',
                'error': 'API key not configured or using placeholder',
                'configured': False
            }
        
        try:
            url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'Global Quote' in data:
                            return {
                                'status': 'SUCCESS',
                                'configured': True,
                                'symbol_tested': 'AAPL',
                                'quote_available': True
                            }
                        elif 'Error Message' in data:
                            return {
                                'status': 'FAILED',
                                'error': data['Error Message'],
                                'configured': True
                            }
                        else:
                            return {
                                'status': 'FAILED',
                                'error': 'Unexpected response format',
                                'configured': True,
                                'response': data
                            }
                    else:
                        return {
                            'status': 'FAILED',
                            'error': f'HTTP {response.status}',
                            'configured': True
                        }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'configured': True
            }
    
    async def test_twelvedata_api(self) -> Dict[str, Any]:
        """Test TwelveData API connectivity and configuration."""
        api_key = os.getenv('TWELVEDATA_API_KEY')
        if not api_key or api_key == 'your_twelvedata_api_key_here':
            return {
                'status': 'FAILED',
                'error': 'API key not configured or using placeholder',
                'configured': False
            }
        
        try:
            url = f'https://api.twelvedata.com/quote?symbol=AAPL&apikey={api_key}'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'symbol' in data:
                            return {
                                'status': 'SUCCESS',
                                'configured': True,
                                'symbol_tested': data.get('symbol'),
                                'price': data.get('close')
                            }
                        else:
                            return {
                                'status': 'FAILED',
                                'error': data.get('message', 'Unknown error'),
                                'configured': True
                            }
                    else:
                        return {
                            'status': 'FAILED',
                            'error': f'HTTP {response.status}',
                            'configured': True
                        }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'configured': True
            }
    
    def test_ibkr_configuration(self) -> Dict[str, Any]:
        """Test IBKR configuration (connection test requires TWS/Gateway running)."""
        host = os.getenv('IBKR_HOST', 'localhost')
        port = os.getenv('IBKR_PORT', '7497')  # Paper trading port
        client_id = os.getenv('IBKR_CLIENT_ID', '1')
        
        try:
            port = int(port)
            client_id = int(client_id)
            
            return {
                'status': 'CONFIGURED',
                'configured': True,
                'host': host,
                'port': port,
                'client_id': client_id,
                'trading_mode': 'paper' if port == 7497 else 'live',
                'note': 'Configuration valid. Connection test requires TWS/Gateway running.'
            }
        except ValueError as e:
            return {
                'status': 'FAILED',
                'error': f'Invalid configuration: {str(e)}',
                'configured': False
            }
    
    async def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive API configuration audit."""
        print("🔍 Starting Comprehensive API Configuration Audit...")
        print("=" * 70)
        
        start_time = time.time()
        
        # Test all APIs
        tests = {
            'OpenAI': self.test_openai_api(),
            'Anthropic': self.test_anthropic_api(),
            'Polygon': self.test_polygon_api(),
            'Alpaca': self.test_alpaca_api(),
            'Alpha Vantage': self.test_alpha_vantage_api(),
            'TwelveData': self.test_twelvedata_api()
        }
        
        # Run async tests
        results = {}
        for service, test_coro in tests.items():
            print(f"Testing {service} API...")
            try:
                result = await test_coro
                results[service] = result
                
                if result['status'] == 'SUCCESS':
                    print(f"✅ {service}: {result['status']}")
                elif result['status'] == 'CONFIGURED':
                    print(f"⚙️ {service}: {result['status']}")
                else:
                    print(f"❌ {service}: {result['status']} - {result.get('error', 'Unknown error')}")
            except Exception as e:
                results[service] = {
                    'status': 'FAILED',
                    'error': f'Test execution failed: {str(e)}',
                    'configured': False
                }
                print(f"❌ {service}: Test execution failed - {str(e)}")
        
        # Test IBKR configuration (synchronous)
        print("Testing IBKR configuration...")
        results['IBKR'] = self.test_ibkr_configuration()
        if results['IBKR']['status'] == 'CONFIGURED':
            print(f"⚙️ IBKR: {results['IBKR']['status']}")
        else:
            print(f"❌ IBKR: {results['IBKR']['status']} - {results['IBKR'].get('error', 'Unknown error')}")
        
        # Calculate summary statistics
        total_services = len(results)
        successful = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
        configured = sum(1 for r in results.values() if r['status'] in ['SUCCESS', 'CONFIGURED'])
        failed = sum(1 for r in results.values() if r['status'] == 'FAILED')
        
        audit_time = time.time() - start_time
        
        summary = {
            'audit_timestamp': datetime.now().isoformat(),
            'total_services': total_services,
            'successful_connections': successful,
            'configured_services': configured,
            'failed_services': failed,
            'success_rate': (successful / total_services) * 100,
            'configuration_rate': (configured / total_services) * 100,
            'audit_duration_seconds': round(audit_time, 2),
            'service_results': results
        }
        
        print("\n" + "=" * 70)
        print("📊 API CONFIGURATION AUDIT SUMMARY")
        print("=" * 70)
        print(f"Total Services Tested: {total_services}")
        print(f"Successful Connections: {successful}")
        print(f"Configured Services: {configured}")
        print(f"Failed Services: {failed}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Configuration Rate: {summary['configuration_rate']:.1f}%")
        print(f"Audit Duration: {audit_time:.2f} seconds")
        
        return summary

async def main():
    """Main execution function."""
    auditor = APIConfigurationAuditor()
    results = await auditor.run_comprehensive_audit()
    
    # Save results to file
    output_file = Path(__file__).parent.parent / "api_configuration_audit_report.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed audit report saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
