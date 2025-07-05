#!/usr/bin/env python3
"""
System Architecture Validation Script for QuantAI.

This script performs comprehensive validation of the system architecture,
including agent discovery, message routing, component integration,
and AutoGen framework compliance.
"""

import asyncio
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import core modules directly to avoid agent import issues
try:
    from quantai.core.config import QuantConfig, get_config
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Config import failed: {e}")
    CONFIG_AVAILABLE = False

try:
    from quantai.core.base import AgentRole, BaseQuantAgent
    BASE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Base import failed: {e}")
    BASE_AVAILABLE = False

try:
    from quantai.core.messages import MessageType, QuantMessage
    MESSAGES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Messages import failed: {e}")
    MESSAGES_AVAILABLE = False

# Import runtime and agents with error handling
try:
    from quantai.core.runtime import QuantRuntime
    RUNTIME_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Runtime import failed: {e}")
    RUNTIME_AVAILABLE = False


class SystemArchitectureValidator:
    """Comprehensive system architecture validator."""
    
    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete system architecture validation."""
        
        print("🔍 Starting QuantAI System Architecture Validation...")
        print("=" * 60)
        
        # Run all validation checks
        await self._validate_agent_architecture()
        await self._validate_message_system()
        await self._validate_autogen_compliance()
        await self._validate_component_integration()
        await self._validate_runtime_system()
        
        # Generate summary report
        return self._generate_validation_report()
    
    async def _validate_agent_architecture(self):
        """Validate agent architecture and organization."""

        print("\n📋 Validating Agent Architecture...")

        if not BASE_AVAILABLE:
            self.errors.append("Cannot validate agent architecture - base module not available")
            print("❌ Cannot validate agent architecture - base module not available")
            return

        try:
            # Check agent role definitions
            expected_roles = {
                'data_layer': [AgentRole.DATA_INGESTION, AgentRole.MULTIMODAL_FUSION],
                'analysis_layer': [AgentRole.MACRO_INSIGHT, AgentRole.STRATEGY_GENERATION, AgentRole.STRATEGY_CODING],
                'validation_layer': [AgentRole.STRATEGY_VALIDATION, AgentRole.RISK_CONTROL],
                'execution_layer': [AgentRole.STRATEGY_DEPLOYMENT, AgentRole.EXECUTION, AgentRole.BACKTEST_MONITOR],
                'learning_layer': [AgentRole.PROFITABILITY, AgentRole.FEEDBACK_LOOP, AgentRole.MEMORY],
                'control_layer': [AgentRole.API_MANAGER, AgentRole.KILL_SWITCH, AgentRole.DASHBOARD]
            }
            
            # Verify all roles exist
            defined_roles = list(AgentRole)
            missing_roles = []
            
            for layer, roles in expected_roles.items():
                for role in roles:
                    if role not in defined_roles:
                        missing_roles.append(f"{role} (from {layer})")
            
            if missing_roles:
                self.errors.append(f"Missing agent roles: {', '.join(missing_roles)}")
            else:
                print("✅ All expected agent roles are defined")
            
            # Check agent implementations
            agent_implementations = self._discover_agent_implementations()
            
            print(f"📊 Found {len(agent_implementations)} agent implementations:")
            for role, agent_class in agent_implementations.items():
                print(f"   • {role.value}: {agent_class.__name__}")
            
            self.validation_results['agent_architecture'] = {
                'total_roles': len(defined_roles),
                'implemented_agents': len(agent_implementations),
                'missing_roles': missing_roles,
                'agent_implementations': {role.value: cls.__name__ for role, cls in agent_implementations.items()}
            }
            
        except Exception as e:
            self.errors.append(f"Agent architecture validation failed: {str(e)}")
            print(f"❌ Agent architecture validation failed: {str(e)}")
    
    async def _validate_message_system(self):
        """Validate message system and routing."""

        print("\n📨 Validating Message System...")

        if not MESSAGES_AVAILABLE:
            self.errors.append("Cannot validate message system - messages module not available")
            print("❌ Cannot validate message system - messages module not available")
            return

        try:
            # Check message type definitions
            message_types = list(MessageType)
            
            # Expected message categories
            expected_categories = {
                'data_messages': ['DATA_REQUEST', 'DATA_RESPONSE'],
                'strategy_messages': ['STRATEGY_GENERATION', 'STRATEGY_CODE', 'STRATEGY_VALIDATION'],
                'trading_messages': ['TRADE_SIGNAL', 'TRADE_REQUEST', 'TRADE_RESPONSE', 'EXECUTION_RESULT'],
                'risk_messages': ['RISK_ASSESSMENT'],
                'backtest_messages': ['BACKTEST_REQUEST', 'BACKTEST_RESPONSE', 'BACKTEST_RESULT'],
                'compliance_messages': ['COMPLIANCE_CHECK_REQUEST', 'AUDIT_LOG_REQUEST', 'TRADE_REPORT_REQUEST'],
                'control_messages': ['KILL_SWITCH', 'SYSTEM_STATUS']
            }
            
            # Check message type coverage
            defined_types = [msg_type.value.upper() for msg_type in message_types]
            missing_types = []
            
            for category, types in expected_categories.items():
                for msg_type in types:
                    if msg_type not in defined_types:
                        missing_types.append(f"{msg_type} (from {category})")
            
            if missing_types:
                self.warnings.append(f"Missing message types: {', '.join(missing_types)}")
            else:
                print("✅ All expected message types are defined")
            
            print(f"📊 Found {len(message_types)} message types")
            
            self.validation_results['message_system'] = {
                'total_message_types': len(message_types),
                'missing_types': missing_types,
                'message_types': [mt.value for mt in message_types]
            }
            
        except Exception as e:
            self.errors.append(f"Message system validation failed: {str(e)}")
            print(f"❌ Message system validation failed: {str(e)}")
    
    async def _validate_autogen_compliance(self):
        """Validate AutoGen framework compliance."""
        
        print("\n🔧 Validating AutoGen Framework Compliance...")
        
        try:
            # Check AutoGen imports
            from autogen_core import RoutedAgent, SingleThreadedAgentRuntime, message_handler, default_subscription
            from autogen_core.models import ChatCompletionClient
            
            print("✅ AutoGen core imports successful")
            
            # Check BaseQuantAgent inheritance
            from quantai.core.base import BaseQuantAgent
            
            if not issubclass(BaseQuantAgent, RoutedAgent):
                self.errors.append("BaseQuantAgent does not inherit from RoutedAgent")
            else:
                print("✅ BaseQuantAgent properly inherits from RoutedAgent")
            
            # Check message handler patterns
            agent_implementations = self._discover_agent_implementations()
            agents_with_handlers = 0
            
            for role, agent_class in agent_implementations.items():
                handler_methods = [
                    method for method in dir(agent_class)
                    if hasattr(getattr(agent_class, method, None), '_message_handler')
                ]
                if handler_methods:
                    agents_with_handlers += 1
            
            print(f"📊 {agents_with_handlers}/{len(agent_implementations)} agents have message handlers")
            
            self.validation_results['autogen_compliance'] = {
                'base_agent_inheritance': issubclass(BaseQuantAgent, RoutedAgent),
                'agents_with_handlers': agents_with_handlers,
                'total_agents': len(agent_implementations)
            }
            
        except ImportError as e:
            self.errors.append(f"AutoGen import failed: {str(e)}")
            print(f"❌ AutoGen import failed: {str(e)}")
        except Exception as e:
            self.errors.append(f"AutoGen compliance validation failed: {str(e)}")
            print(f"❌ AutoGen compliance validation failed: {str(e)}")
    
    async def _validate_component_integration(self):
        """Validate integration of new components."""
        
        print("\n🔗 Validating Component Integration...")
        
        try:
            # Test compliance components
            compliance_components = [
                'quantai.compliance.audit_trail.AuditTrailManager',
                'quantai.compliance.trade_reporting.TradeReportingEngine',
                'quantai.compliance.compliance_monitor.ComplianceMonitor',
                'quantai.compliance.regulatory_reporting.RegulatoryReportingGenerator'
            ]
            
            compliance_ok = self._test_component_imports(compliance_components, "Compliance")
            
            # Test analytics components
            analytics_components = [
                'quantai.analytics.advanced_risk.AdvancedRiskAnalytics',
                'quantai.analytics.performance_attribution.PerformanceAttributionEngine',
                'quantai.core.account_manager.MultiAccountManager'
            ]
            
            analytics_ok = self._test_component_imports(analytics_components, "Analytics")
            
            # Test agent integrations
            agent_components = [
                'quantai.agents.compliance.compliance_agent.ComplianceAgent'
            ]
            
            agents_ok = self._test_component_imports(agent_components, "Agent Integration")
            
            self.validation_results['component_integration'] = {
                'compliance_components': compliance_ok,
                'analytics_components': analytics_ok,
                'agent_components': agents_ok
            }
            
        except Exception as e:
            self.errors.append(f"Component integration validation failed: {str(e)}")
            print(f"❌ Component integration validation failed: {str(e)}")
    
    async def _validate_runtime_system(self):
        """Validate runtime system functionality."""

        print("\n⚙️ Validating Runtime System...")

        if not RUNTIME_AVAILABLE:
            self.warnings.append("Runtime system not available due to missing dependencies")
            print("⚠️ Runtime system not available due to missing dependencies")
            self.validation_results['runtime_system'] = {
                'runtime_available': False,
                'runtime_init': False,
                'registry_accessible': False,
                'router_accessible': False
            }
            return

        try:
            # Test runtime initialization
            config = QuantConfig(environment="test")
            runtime = QuantRuntime(config)

            print("✅ Runtime initialization successful")

            # Test agent registry
            registry = runtime.registry
            print("✅ Agent registry accessible")

            # Test message router
            router = runtime.router
            print("✅ Message router accessible")

            self.validation_results['runtime_system'] = {
                'runtime_available': True,
                'runtime_init': True,
                'registry_accessible': True,
                'router_accessible': True
            }

        except Exception as e:
            self.errors.append(f"Runtime system validation failed: {str(e)}")
            print(f"❌ Runtime system validation failed: {str(e)}")
    
    def _discover_agent_implementations(self) -> Dict[AgentRole, type]:
        """Discover available agent implementations."""

        agent_implementations = {}

        # Define all expected agents and their import paths
        agent_imports = {
            AgentRole.DATA_INGESTION: ("quantai.agents.data.ingestion", "DataIngestionAgent"),
            AgentRole.MULTIMODAL_FUSION: ("quantai.agents.data.fusion", "MultimodalFusionAgent"),
            AgentRole.MACRO_INSIGHT: ("quantai.agents.analysis.macro", "MacroInsightAgent"),
            AgentRole.STRATEGY_GENERATION: ("quantai.agents.analysis.strategy_generation", "StrategyGenerationAgent"),
            AgentRole.STRATEGY_CODING: ("quantai.agents.analysis.strategy_coding", "StrategyCodingAgent"),
            AgentRole.STRATEGY_VALIDATION: ("quantai.agents.validation.strategy_validator", "StrategyValidationAgent"),
            AgentRole.RISK_CONTROL: ("quantai.agents.validation.risk_control", "RiskControlAgent"),
            AgentRole.STRATEGY_DEPLOYMENT: ("quantai.agents.execution.deployment", "StrategyDeploymentAgent"),
            AgentRole.EXECUTION: ("quantai.agents.execution.trader", "ExecutionAgent"),
            AgentRole.BACKTEST_MONITOR: ("quantai.agents.execution.monitor", "BacktestMonitorAgent"),
            AgentRole.PROFITABILITY: ("quantai.agents.learning.profitability", "ProfitabilityAgent"),
            AgentRole.FEEDBACK_LOOP: ("quantai.agents.learning.feedback_loop", "FeedbackLoopAgent"),
            AgentRole.MEMORY: ("quantai.agents.learning.memory", "MemoryAgent"),
            AgentRole.API_MANAGER: ("quantai.agents.control.api_manager", "APIManagerAgent"),
            AgentRole.KILL_SWITCH: ("quantai.agents.control.kill_switch", "KillSwitchAgent"),
            AgentRole.DASHBOARD: ("quantai.agents.control.dashboard", "DashboardAgent"),
        }

        # Try to import each agent
        for role, (module_path, class_name) in agent_imports.items():
            try:
                module = __import__(module_path, fromlist=[class_name])
                agent_class = getattr(module, class_name)
                agent_implementations[role] = agent_class
                print(f"✅ Successfully imported {role.value}: {class_name}")
            except ImportError as e:
                print(f"⚠️ Failed to import {role.value}: {e}")
            except AttributeError as e:
                print(f"⚠️ Class {class_name} not found in {module_path}: {e}")

        return agent_implementations
    
    def _test_component_imports(self, components: List[str], component_type: str) -> bool:
        """Test if components can be imported successfully."""
        
        success_count = 0
        
        for component in components:
            try:
                module_path, class_name = component.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                getattr(module, class_name)
                success_count += 1
            except ImportError as e:
                self.warnings.append(f"Failed to import {component}: {str(e)}")
        
        success_rate = success_count / len(components)
        
        if success_rate == 1.0:
            print(f"✅ {component_type} components: All {len(components)} imported successfully")
        elif success_rate >= 0.8:
            print(f"⚠️ {component_type} components: {success_count}/{len(components)} imported successfully")
        else:
            print(f"❌ {component_type} components: Only {success_count}/{len(components)} imported successfully")
        
        return success_rate >= 0.8
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        # Calculate overall status
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)
        
        if total_errors == 0 and total_warnings == 0:
            status = "PASS"
            print("🎉 System architecture validation: PASSED")
        elif total_errors == 0:
            status = "PASS_WITH_WARNINGS"
            print(f"⚠️ System architecture validation: PASSED with {total_warnings} warnings")
        else:
            status = "FAIL"
            print(f"❌ System architecture validation: FAILED with {total_errors} errors")
        
        # Print errors and warnings
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        # Generate report
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': status,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'errors': self.errors,
            'warnings': self.warnings,
            'validation_results': self.validation_results
        }
        
        return report


async def main():
    """Main validation function."""
    
    validator = SystemArchitectureValidator()
    
    try:
        report = await validator.run_full_validation()
        
        # Save report to file
        import json
        report_file = Path("validation_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        if report['overall_status'] == 'FAIL':
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Validation failed with exception: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
