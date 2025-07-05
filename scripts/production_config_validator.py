#!/usr/bin/env python3
"""
Production Configuration Validator
Validates production deployment configuration and readiness.
"""

import os
import json
import yaml
import subprocess
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import docker
import psutil

def load_env_file(env_file_path: str = '.env'):
    """Load environment variables from .env file."""
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value

class ProductionConfigValidator:
    """Comprehensive production configuration validator."""
    
    def __init__(self):
        # Load environment variables from .env file
        load_env_file()

        self.validation_results = {}
        self.errors = []
        self.warnings = []
        self.docker_client = None

        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.warnings.append(f"Docker client unavailable: {e}")
    
    def validate_environment_variables(self) -> Dict[str, Any]:
        """Validate production environment variables."""
        print("🔧 Validating Production Environment Variables...")
        
        required_vars = {
            'POSTGRES_PASSWORD': 'Database password must be set',
            'QUANTAI_ENV': 'Environment must be set to production',
            'LOG_LEVEL': 'Log level should be WARNING or ERROR for production',
            'TRADING_MODE': 'Trading mode must be explicitly set',
            'IBKR_HOST': 'IBKR host must be configured',
            'IBKR_PORT': 'IBKR port must be configured',
            'IBKR_ACCOUNT': 'IBKR account must be configured'
        }
        
        optional_vars = {
            'MAX_WORKERS': 'Number of worker processes',
            'WORKER_TIMEOUT': 'Worker timeout in seconds',
            'DATA_PATH': 'Custom data path for volumes',
            'SSL_CERT_PATH': 'SSL certificate path',
            'SSL_KEY_PATH': 'SSL private key path'
        }
        
        results = {
            'required_vars': {},
            'optional_vars': {},
            'security_vars': {},
            'performance_vars': {}
        }
        
        # Check required variables
        for var, description in required_vars.items():
            value = os.getenv(var)
            if value:
                results['required_vars'][var] = {'status': 'OK', 'value': value[:10] + '...' if len(value) > 10 else value}
                print(f"   ✅ {var}: Configured")
            else:
                results['required_vars'][var] = {'status': 'MISSING', 'description': description}
                self.errors.append(f"Missing required environment variable: {var}")
                print(f"   ❌ {var}: Missing - {description}")
        
        # Check optional variables
        for var, description in optional_vars.items():
            value = os.getenv(var)
            if value:
                results['optional_vars'][var] = {'status': 'OK', 'value': value}
                print(f"   ✅ {var}: {value}")
            else:
                results['optional_vars'][var] = {'status': 'DEFAULT', 'description': description}
                print(f"   ⚠️ {var}: Using default - {description}")
        
        # Validate specific values
        if os.getenv('QUANTAI_ENV') != 'production':
            self.warnings.append("QUANTAI_ENV should be 'production' for production deployment")
        
        if os.getenv('LOG_LEVEL') not in ['WARNING', 'ERROR']:
            self.warnings.append("LOG_LEVEL should be WARNING or ERROR for production")
        
        if os.getenv('TRADING_MODE') != 'paper':
            self.warnings.append("TRADING_MODE should be 'paper' for safety")
        
        return results
    
    def validate_docker_configuration(self) -> Dict[str, Any]:
        """Validate Docker production configuration."""
        print("🐳 Validating Docker Production Configuration...")
        
        results = {
            'compose_files': {},
            'dockerfiles': {},
            'resource_limits': {},
            'health_checks': {},
            'security_settings': {}
        }
        
        # Check Docker Compose files
        compose_files = ['docker-compose.yml', 'docker-compose.prod.yml']
        for file in compose_files:
            if os.path.exists(file):
                try:
                    with open(file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    results['compose_files'][file] = {
                        'status': 'OK',
                        'services': len(config.get('services', {})),
                        'volumes': len(config.get('volumes', {})),
                        'networks': len(config.get('networks', {}))
                    }
                    print(f"   ✅ {file}: Valid ({len(config.get('services', {}))} services)")
                    
                    # Check for production optimizations
                    if file == 'docker-compose.prod.yml':
                        self._validate_production_optimizations(config, results)
                        
                except Exception as e:
                    results['compose_files'][file] = {'status': 'ERROR', 'error': str(e)}
                    self.errors.append(f"Invalid Docker Compose file {file}: {e}")
                    print(f"   ❌ {file}: Invalid - {e}")
            else:
                results['compose_files'][file] = {'status': 'MISSING'}
                if file == 'docker-compose.prod.yml':
                    self.errors.append(f"Missing production Docker Compose file: {file}")
                    print(f"   ❌ {file}: Missing")
                else:
                    print(f"   ⚠️ {file}: Missing (optional)")
        
        return results
    
    def _validate_production_optimizations(self, config: Dict, results: Dict):
        """Validate production-specific optimizations in Docker config."""
        services = config.get('services', {})
        
        for service_name, service_config in services.items():
            # Check resource limits
            deploy = service_config.get('deploy', {})
            resources = deploy.get('resources', {})
            limits = resources.get('limits', {})
            
            if limits:
                results['resource_limits'][service_name] = {
                    'memory': limits.get('memory', 'Not set'),
                    'cpus': limits.get('cpus', 'Not set')
                }
                print(f"   ✅ {service_name}: Resource limits configured")
            else:
                self.warnings.append(f"Service {service_name} missing resource limits")
                print(f"   ⚠️ {service_name}: No resource limits")
            
            # Check health checks
            healthcheck = service_config.get('healthcheck')
            if healthcheck:
                results['health_checks'][service_name] = {
                    'test': healthcheck.get('test', 'Not specified'),
                    'interval': healthcheck.get('interval', 'Default'),
                    'timeout': healthcheck.get('timeout', 'Default'),
                    'retries': healthcheck.get('retries', 'Default')
                }
                print(f"   ✅ {service_name}: Health check configured")
            else:
                self.warnings.append(f"Service {service_name} missing health check")
                print(f"   ⚠️ {service_name}: No health check")
            
            # Check security settings
            security_opt = service_config.get('security_opt', [])
            if security_opt:
                results['security_settings'][service_name] = security_opt
                print(f"   ✅ {service_name}: Security options configured")
            else:
                self.warnings.append(f"Service {service_name} missing security options")
                print(f"   ⚠️ {service_name}: No security options")
    
    def validate_system_resources(self) -> Dict[str, Any]:
        """Validate system resources for production deployment."""
        print("💻 Validating System Resources...")
        
        # Get system information
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        results = {
            'cpu': {
                'cores': cpu_count,
                'usage_percent': psutil.cpu_percent(interval=1),
                'recommendation': 'OK' if cpu_count >= 4 else 'UPGRADE_RECOMMENDED'
            },
            'memory': {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'usage_percent': memory.percent,
                'recommendation': 'OK' if memory.total >= 8 * (1024**3) else 'UPGRADE_RECOMMENDED'
            },
            'disk': {
                'total_gb': round(disk.total / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'usage_percent': round((disk.used / disk.total) * 100, 2),
                'recommendation': 'OK' if disk.free >= 50 * (1024**3) else 'CLEANUP_NEEDED'
            }
        }
        
        print(f"   💾 CPU: {cpu_count} cores ({psutil.cpu_percent(interval=1)}% usage)")
        print(f"   🧠 Memory: {results['memory']['total_gb']}GB total, {results['memory']['available_gb']}GB available")
        print(f"   💽 Disk: {results['disk']['total_gb']}GB total, {results['disk']['free_gb']}GB free")
        
        # Check recommendations
        if results['cpu']['recommendation'] != 'OK':
            self.warnings.append(f"CPU: {cpu_count} cores may be insufficient for production (recommended: 4+)")
        
        if results['memory']['recommendation'] != 'OK':
            self.warnings.append(f"Memory: {results['memory']['total_gb']}GB may be insufficient for production (recommended: 8GB+)")
        
        if results['disk']['recommendation'] != 'OK':
            self.warnings.append(f"Disk: {results['disk']['free_gb']}GB free space may be insufficient (recommended: 50GB+)")
        
        return results
    
    def validate_network_connectivity(self) -> Dict[str, Any]:
        """Validate network connectivity and port availability."""
        print("🌐 Validating Network Configuration...")
        
        required_ports = {
            80: 'HTTP (Nginx)',
            443: 'HTTPS (Nginx)',
            8000: 'QuantAI API',
            5432: 'PostgreSQL',
            6379: 'Redis'
        }
        
        results = {
            'port_availability': {},
            'external_connectivity': {}
        }
        
        # Check port availability
        for port, description in required_ports.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    results['port_availability'][port] = {'status': 'IN_USE', 'description': description}
                    print(f"   ⚠️ Port {port}: In use ({description})")
                else:
                    results['port_availability'][port] = {'status': 'AVAILABLE', 'description': description}
                    print(f"   ✅ Port {port}: Available ({description})")
            except Exception as e:
                results['port_availability'][port] = {'status': 'ERROR', 'error': str(e)}
                self.errors.append(f"Error checking port {port}: {e}")
        
        return results
    
    def validate_ssl_configuration(self) -> Dict[str, Any]:
        """Validate SSL/TLS configuration."""
        print("🔒 Validating SSL Configuration...")
        
        ssl_paths = {
            'cert': os.getenv('SSL_CERT_PATH', './nginx/ssl/cert.pem'),
            'key': os.getenv('SSL_KEY_PATH', './nginx/ssl/key.pem')
        }
        
        results = {
            'ssl_files': {},
            'configuration': {}
        }
        
        for file_type, path in ssl_paths.items():
            if os.path.exists(path):
                results['ssl_files'][file_type] = {'status': 'OK', 'path': path}
                print(f"   ✅ SSL {file_type}: Found at {path}")
            else:
                results['ssl_files'][file_type] = {'status': 'MISSING', 'path': path}
                self.warnings.append(f"SSL {file_type} file missing: {path}")
                print(f"   ⚠️ SSL {file_type}: Missing at {path}")
        
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive production configuration validation."""
        print("🚀 Starting Production Configuration Validation...")
        print("=" * 70)
        
        start_time = time.time()
        
        # Run all validations
        validations = {
            'environment_variables': self.validate_environment_variables(),
            'docker_configuration': self.validate_docker_configuration(),
            'system_resources': self.validate_system_resources(),
            'network_connectivity': self.validate_network_connectivity(),
            'ssl_configuration': self.validate_ssl_configuration()
        }
        
        # Calculate overall score
        total_checks = 0
        passed_checks = 0
        
        for category, results in validations.items():
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            total_checks += 1
                            if isinstance(subvalue, dict) and subvalue.get('status') in ['OK', 'AVAILABLE']:
                                passed_checks += 1
        
        validation_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        # Summary
        summary = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_duration': round(time.time() - start_time, 2),
            'validation_score': round(validation_score, 1),
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'error_list': self.errors,
            'warning_list': self.warnings,
            'validations': validations,
            'production_ready': len(self.errors) == 0 and validation_score >= 90
        }
        
        print("\n" + "=" * 70)
        print("📊 PRODUCTION CONFIGURATION VALIDATION SUMMARY")
        print("=" * 70)
        print(f"🎯 Validation Score: {validation_score:.1f}%")
        print(f"✅ Passed Checks: {passed_checks}/{total_checks}")
        print(f"❌ Errors: {len(self.errors)}")
        print(f"⚠️ Warnings: {len(self.warnings)}")
        print(f"🚀 Production Ready: {'YES' if summary['production_ready'] else 'NO'}")
        
        if self.errors:
            print(f"\n❌ Critical Errors:")
            for error in self.errors:
                print(f"   - {error}")
        
        if self.warnings:
            print(f"\n⚠️ Warnings:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        return summary

def main():
    """Main execution function."""
    validator = ProductionConfigValidator()
    results = validator.run_comprehensive_validation()
    
    # Save results
    output_file = Path(__file__).parent.parent / "production_config_validation_report.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed validation report saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()
