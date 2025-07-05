#!/usr/bin/env python3
"""
Pre-Deployment Verification Script
Comprehensive checks before production deployment.
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

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

class PreDeploymentChecker:
    """Comprehensive pre-deployment verification."""
    
    def __init__(self):
        # Load environment variables from .env file
        load_env_file()

        self.checks_passed = 0
        self.checks_failed = 0
        self.critical_failures = []
        self.warnings = []
        
    def check_environment_setup(self) -> bool:
        """Check environment setup and configuration."""
        print("🔧 Checking Environment Setup...")
        
        # Check .env file
        if not os.path.exists('.env'):
            self.critical_failures.append("Missing .env file")
            print("   ❌ .env file: Missing")
            return False
        
        # Check critical environment variables
        critical_vars = [
            'POSTGRES_PASSWORD', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
            'IBKR_HOST', 'IBKR_PORT', 'IBKR_ACCOUNT', 'TRADING_MODE'
        ]
        
        missing_vars = []
        for var in critical_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.critical_failures.append(f"Missing environment variables: {', '.join(missing_vars)}")
            print(f"   ❌ Environment variables: Missing {len(missing_vars)} critical variables")
            return False
        
        print("   ✅ Environment setup: OK")
        self.checks_passed += 1
        return True
    
    def check_docker_setup(self) -> bool:
        """Check Docker setup and configuration."""
        print("🐳 Checking Docker Setup...")
        
        try:
            # Check Docker is running
            result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
            if result.returncode != 0:
                self.critical_failures.append("Docker is not running")
                print("   ❌ Docker: Not running")
                return False
            
            # Check Docker Compose
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                self.critical_failures.append("Docker Compose not available")
                print("   ❌ Docker Compose: Not available")
                return False
            
            # Check production compose file
            if not os.path.exists('docker-compose.prod.yml'):
                self.critical_failures.append("Missing docker-compose.prod.yml")
                print("   ❌ Production compose file: Missing")
                return False
            
            print("   ✅ Docker setup: OK")
            self.checks_passed += 1
            return True
            
        except Exception as e:
            self.critical_failures.append(f"Docker check failed: {e}")
            print(f"   ❌ Docker: Error - {e}")
            return False
    
    def check_system_resources(self) -> bool:
        """Check system resources availability."""
        print("💻 Checking System Resources...")
        
        try:
            import psutil
            
            # Check CPU
            cpu_count = psutil.cpu_count()
            if cpu_count < 2:
                self.warnings.append(f"Low CPU count: {cpu_count} cores (recommended: 4+)")
                print(f"   ⚠️ CPU: {cpu_count} cores (low)")
            else:
                print(f"   ✅ CPU: {cpu_count} cores")
            
            # Check Memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb < 4:
                self.critical_failures.append(f"Insufficient memory: {memory_gb:.1f}GB (minimum: 4GB)")
                print(f"   ❌ Memory: {memory_gb:.1f}GB (insufficient)")
                return False
            elif memory_gb < 8:
                self.warnings.append(f"Low memory: {memory_gb:.1f}GB (recommended: 8GB+)")
                print(f"   ⚠️ Memory: {memory_gb:.1f}GB (low)")
            else:
                print(f"   ✅ Memory: {memory_gb:.1f}GB")
            
            # Check Disk Space
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            if disk_free_gb < 10:
                self.critical_failures.append(f"Insufficient disk space: {disk_free_gb:.1f}GB (minimum: 10GB)")
                print(f"   ❌ Disk: {disk_free_gb:.1f}GB free (insufficient)")
                return False
            elif disk_free_gb < 50:
                self.warnings.append(f"Low disk space: {disk_free_gb:.1f}GB (recommended: 50GB+)")
                print(f"   ⚠️ Disk: {disk_free_gb:.1f}GB free (low)")
            else:
                print(f"   ✅ Disk: {disk_free_gb:.1f}GB free")
            
            self.checks_passed += 1
            return True
            
        except ImportError:
            self.warnings.append("psutil not available for system resource checking")
            print("   ⚠️ System resources: Cannot check (psutil missing)")
            return True
        except Exception as e:
            self.warnings.append(f"System resource check failed: {e}")
            print(f"   ⚠️ System resources: Error - {e}")
            return True
    
    def check_dependencies(self) -> bool:
        """Check Python dependencies."""
        print("📦 Checking Dependencies...")
        
        try:
            # Check requirements.txt exists
            if not os.path.exists('requirements.txt'):
                self.critical_failures.append("Missing requirements.txt")
                print("   ❌ requirements.txt: Missing")
                return False
            
            # Try to import critical packages
            critical_packages = [
                'autogen_agentchat', 'fastapi', 'uvicorn', 'sqlalchemy',
                'redis', 'docker', 'psutil', 'pydantic'
            ]
            
            missing_packages = []
            for package in critical_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                self.critical_failures.append(f"Missing packages: {', '.join(missing_packages)}")
                print(f"   ❌ Dependencies: Missing {len(missing_packages)} packages")
                return False
            
            print("   ✅ Dependencies: OK")
            self.checks_passed += 1
            return True
            
        except Exception as e:
            self.critical_failures.append(f"Dependency check failed: {e}")
            print(f"   ❌ Dependencies: Error - {e}")
            return False
    
    def check_configuration_files(self) -> bool:
        """Check configuration files."""
        print("📋 Checking Configuration Files...")
        
        required_files = [
            'docker-compose.yml',
            'docker-compose.prod.yml',
            'Dockerfile',
            'requirements.txt',
            '.env'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            self.critical_failures.append(f"Missing configuration files: {', '.join(missing_files)}")
            print(f"   ❌ Configuration files: Missing {len(missing_files)} files")
            return False
        
        print("   ✅ Configuration files: OK")
        self.checks_passed += 1
        return True
    
    def check_security_settings(self) -> bool:
        """Check security settings."""
        print("🔒 Checking Security Settings...")
        
        # Check trading mode is paper
        trading_mode = os.getenv('TRADING_MODE', '').lower()
        if trading_mode != 'paper':
            self.warnings.append(f"Trading mode is '{trading_mode}' (recommended: 'paper' for safety)")
            print(f"   ⚠️ Trading mode: {trading_mode} (not paper)")
        else:
            print("   ✅ Trading mode: paper (safe)")
        
        # Check for strong passwords
        postgres_password = os.getenv('POSTGRES_PASSWORD', '')
        if len(postgres_password) < 12:
            self.warnings.append("Postgres password should be at least 12 characters")
            print("   ⚠️ Postgres password: Weak")
        else:
            print("   ✅ Postgres password: Strong")
        
        # Check log level
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        if log_level not in ['WARNING', 'ERROR']:
            self.warnings.append(f"Log level is '{log_level}' (recommended: WARNING or ERROR for production)")
            print(f"   ⚠️ Log level: {log_level}")
        else:
            print(f"   ✅ Log level: {log_level}")
        
        self.checks_passed += 1
        return True
    
    def check_api_connectivity(self) -> bool:
        """Check API connectivity."""
        print("🌐 Checking API Connectivity...")
        
        try:
            # Run API configuration audit
            result = subprocess.run([
                'python', 'scripts/api_configuration_audit.py'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse the output to check success rate
                if "Success Rate: 85.7%" in result.stdout or "Success Rate: 100%" in result.stdout:
                    print("   ✅ API connectivity: OK")
                    self.checks_passed += 1
                    return True
                else:
                    self.warnings.append("API connectivity issues detected")
                    print("   ⚠️ API connectivity: Some issues")
                    return True
            else:
                self.warnings.append(f"API connectivity check failed: {result.stderr}")
                print("   ⚠️ API connectivity: Check failed")
                return True
                
        except subprocess.TimeoutExpired:
            self.warnings.append("API connectivity check timed out")
            print("   ⚠️ API connectivity: Timeout")
            return True
        except Exception as e:
            self.warnings.append(f"API connectivity check error: {e}")
            print(f"   ⚠️ API connectivity: Error - {e}")
            return True
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive pre-deployment check."""
        print("🚀 Starting Pre-Deployment Verification...")
        print("=" * 70)
        
        start_time = time.time()
        
        # Run all checks
        checks = [
            ('Environment Setup', self.check_environment_setup),
            ('Docker Setup', self.check_docker_setup),
            ('System Resources', self.check_system_resources),
            ('Dependencies', self.check_dependencies),
            ('Configuration Files', self.check_configuration_files),
            ('Security Settings', self.check_security_settings),
            ('API Connectivity', self.check_api_connectivity)
        ]
        
        total_checks = len(checks)
        
        for check_name, check_func in checks:
            try:
                check_func()
            except Exception as e:
                self.critical_failures.append(f"{check_name} check failed: {e}")
                self.checks_failed += 1
                print(f"   ❌ {check_name}: Unexpected error - {e}")
        
        # Calculate results
        duration = time.time() - start_time
        success_rate = (self.checks_passed / total_checks * 100) if total_checks > 0 else 0
        deployment_ready = len(self.critical_failures) == 0
        
        # Summary
        summary = {
            'check_timestamp': datetime.now().isoformat(),
            'check_duration': round(duration, 2),
            'total_checks': total_checks,
            'checks_passed': self.checks_passed,
            'checks_failed': self.checks_failed,
            'success_rate': round(success_rate, 1),
            'critical_failures': self.critical_failures,
            'warnings': self.warnings,
            'deployment_ready': deployment_ready
        }
        
        print("\n" + "=" * 70)
        print("📊 PRE-DEPLOYMENT VERIFICATION SUMMARY")
        print("=" * 70)
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        print(f"✅ Checks Passed: {self.checks_passed}/{total_checks}")
        print(f"❌ Critical Failures: {len(self.critical_failures)}")
        print(f"⚠️ Warnings: {len(self.warnings)}")
        print(f"🚀 Deployment Ready: {'YES' if deployment_ready else 'NO'}")
        
        if self.critical_failures:
            print(f"\n❌ Critical Issues (Must Fix Before Deployment):")
            for failure in self.critical_failures:
                print(f"   - {failure}")
        
        if self.warnings:
            print(f"\n⚠️ Warnings (Recommended to Address):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if deployment_ready:
            print(f"\n🎉 System is ready for production deployment!")
            print(f"   Run: docker-compose -f docker-compose.prod.yml up -d")
        else:
            print(f"\n🛑 System is NOT ready for deployment. Fix critical issues first.")
        
        return summary

def main():
    """Main execution function."""
    checker = PreDeploymentChecker()
    results = checker.run_comprehensive_check()
    
    # Save results
    output_file = Path(__file__).parent.parent / "pre_deployment_check_report.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed check report saved to: {output_file}")
    
    # Exit with appropriate code
    sys.exit(0 if results['deployment_ready'] else 1)

if __name__ == "__main__":
    main()
