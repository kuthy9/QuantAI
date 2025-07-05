# QuantAI Disaster Recovery Plan

**Document Version:** 1.0  
**Last Updated:** 2025-01-04  
**Review Schedule:** Quarterly

---

## Executive Summary

This document outlines the disaster recovery procedures for the QuantAI quantitative trading system. The plan ensures business continuity, data protection, and rapid system recovery in case of system failures, data corruption, or other catastrophic events.

**Recovery Time Objective (RTO):** 4 hours  
**Recovery Point Objective (RPO):** 15 minutes  
**Maximum Tolerable Downtime:** 8 hours

---

## 1. Backup Strategy

### 1.1 Data Classification

#### Critical Data (Tier 1)
- **Trading positions and orders**
- **Account balances and P&L**
- **Risk management settings**
- **Compliance audit trails**
- **Configuration files**

**Backup Frequency:** Real-time replication + 15-minute snapshots  
**Retention:** 7 years (regulatory requirement)

#### Important Data (Tier 2)
- **Historical market data**
- **Strategy performance metrics**
- **System logs**
- **User preferences**

**Backup Frequency:** Hourly snapshots  
**Retention:** 2 years

#### Standard Data (Tier 3)
- **Cached market data**
- **Temporary files**
- **Development artifacts**

**Backup Frequency:** Daily snapshots  
**Retention:** 30 days

### 1.2 Backup Infrastructure

#### Primary Backup System
- **Location:** Local NAS/SAN storage
- **Technology:** Incremental snapshots with deduplication
- **Schedule:** Continuous for Tier 1, Hourly for Tier 2, Daily for Tier 3
- **Verification:** Automated integrity checks every 4 hours

#### Secondary Backup System
- **Location:** Cloud storage (AWS S3/Azure Blob)
- **Technology:** Encrypted, versioned storage
- **Schedule:** Daily full backup, hourly incremental
- **Verification:** Weekly restore tests

#### Tertiary Backup System
- **Location:** Offsite physical storage
- **Technology:** Encrypted tape/disk storage
- **Schedule:** Weekly full backup
- **Verification:** Monthly restore tests

### 1.3 Backup Procedures

#### Automated Backup Process
```bash
# Daily backup script (runs at 2 AM)
#!/bin/bash
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/quantai/$BACKUP_DATE"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup database
sqlite3 data/quantai.db ".backup $BACKUP_DIR/quantai_$BACKUP_DATE.db"

# Backup configuration
cp -r config/ "$BACKUP_DIR/config/"
cp .env "$BACKUP_DIR/env_$BACKUP_DATE"

# Backup logs
cp -r logs/ "$BACKUP_DIR/logs/"

# Compress and encrypt
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
gpg --cipher-algo AES256 --compress-algo 1 --symmetric \
    --output "$BACKUP_DIR.tar.gz.gpg" "$BACKUP_DIR.tar.gz"

# Upload to cloud
aws s3 cp "$BACKUP_DIR.tar.gz.gpg" s3://quantai-backups/daily/

# Cleanup local files older than 30 days
find /backups/quantai/ -name "*.tar.gz.gpg" -mtime +30 -delete
```

#### Manual Backup Checklist
- [ ] Stop all trading activities
- [ ] Export current positions and orders
- [ ] Backup database with integrity check
- [ ] Copy all configuration files
- [ ] Export system logs
- [ ] Verify backup completeness
- [ ] Test restore procedure
- [ ] Document backup location and encryption keys

---

## 2. Disaster Recovery Procedures

### 2.1 Incident Classification

#### Severity 1 - Critical
- **Complete system failure**
- **Data corruption affecting trading**
- **Security breach with data compromise**
- **Regulatory compliance violation**

**Response Time:** Immediate (< 15 minutes)  
**Escalation:** CEO, CTO, Compliance Officer

#### Severity 2 - High
- **Partial system failure**
- **Performance degradation affecting trading**
- **Non-critical data corruption**
- **API service outages**

**Response Time:** 1 hour  
**Escalation:** Technical Lead, Operations Manager

#### Severity 3 - Medium
- **Minor system issues**
- **Non-trading system failures**
- **Scheduled maintenance overruns**

**Response Time:** 4 hours  
**Escalation:** Development Team

### 2.2 Recovery Procedures

#### Complete System Recovery
1. **Immediate Actions (0-15 minutes)**
   - Activate emergency kill switch
   - Stop all automated trading
   - Notify key stakeholders
   - Assess scope of failure
   - Activate backup systems if available

2. **Assessment Phase (15-60 minutes)**
   - Determine root cause
   - Assess data integrity
   - Evaluate recovery options
   - Estimate recovery time
   - Communicate status to stakeholders

3. **Recovery Phase (1-4 hours)**
   - Provision new infrastructure if needed
   - Restore from most recent backup
   - Verify data integrity
   - Test critical system functions
   - Validate trading connectivity

4. **Validation Phase (4-6 hours)**
   - Run comprehensive system tests
   - Verify all agent functionality
   - Test trading workflows
   - Validate compliance systems
   - Perform reconciliation

5. **Resume Operations (6-8 hours)**
   - Gradually resume trading activities
   - Monitor system performance
   - Document lessons learned
   - Update recovery procedures

#### Database Recovery
```sql
-- Verify database integrity
PRAGMA integrity_check;

-- Restore from backup
.restore /backups/quantai/latest/quantai.db

-- Verify critical tables
SELECT COUNT(*) FROM trades;
SELECT COUNT(*) FROM positions;
SELECT COUNT(*) FROM audit_trail;

-- Check data consistency
SELECT * FROM positions WHERE quantity != 0;
SELECT * FROM trades WHERE status = 'pending';
```

#### Configuration Recovery
```bash
# Restore configuration files
cp /backups/quantai/latest/config/* config/
cp /backups/quantai/latest/env_* .env

# Verify configuration
python -c "from quantai.core.config import QuantConfig; print('Config OK')"

# Test API connections
python scripts/test_connections.py
```

### 2.3 Failover Procedures

#### Primary to Secondary System
1. **Automatic Failover (< 5 minutes)**
   - Health monitoring detects failure
   - DNS switches to secondary system
   - Load balancer redirects traffic
   - Secondary system activates

2. **Manual Failover (< 15 minutes)**
   - Operations team initiates failover
   - Update DNS records
   - Activate secondary infrastructure
   - Verify system functionality

#### Trading Platform Failover
1. **IBKR Primary → IBKR Backup**
   - Switch to backup IBKR connection
   - Verify account access
   - Reconcile positions
   - Resume trading

2. **IBKR → Alpaca Failover**
   - Export positions from IBKR
   - Calculate equivalent positions in Alpaca
   - Execute hedge trades if necessary
   - Update system configuration

---

## 3. Data Recovery Procedures

### 3.1 Database Recovery

#### SQLite Database Recovery
```bash
# Check database integrity
sqlite3 data/quantai.db "PRAGMA integrity_check;"

# Recover from corruption
sqlite3 data/quantai.db ".recover" | sqlite3 data/quantai_recovered.db

# Restore from backup
cp /backups/quantai/latest/quantai.db data/quantai.db
```

#### Data Validation After Recovery
```python
# Validate critical data integrity
import sqlite3

def validate_data_integrity():
    conn = sqlite3.connect('data/quantai.db')
    
    # Check for orphaned records
    orphaned_trades = conn.execute("""
        SELECT COUNT(*) FROM trades t 
        LEFT JOIN positions p ON t.position_id = p.id 
        WHERE p.id IS NULL
    """).fetchone()[0]
    
    # Check for negative balances
    negative_balances = conn.execute("""
        SELECT COUNT(*) FROM accounts 
        WHERE balance < 0 AND account_type != 'margin'
    """).fetchone()[0]
    
    # Verify audit trail continuity
    audit_gaps = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT id, LAG(id) OVER (ORDER BY timestamp) as prev_id
            FROM audit_trail
        ) WHERE id - prev_id > 1
    """).fetchone()[0]
    
    return {
        'orphaned_trades': orphaned_trades,
        'negative_balances': negative_balances,
        'audit_gaps': audit_gaps
    }
```

### 3.2 Configuration Recovery

#### Environment Configuration
```bash
# Restore environment variables
cp /backups/quantai/latest/.env .env

# Validate configuration
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

required_vars = [
    'OPENAI_API_KEY', 'IBKR_HOST', 'IBKR_PORT',
    'ALPACA_API_KEY', 'TRADING_MODE'
]

missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    print(f'Missing variables: {missing}')
else:
    print('Configuration valid')
"
```

#### Agent Configuration Recovery
```python
# Restore agent configurations
import json
import shutil

def restore_agent_configs():
    backup_path = "/backups/quantai/latest/config/agents/"
    config_path = "config/agents/"
    
    # Copy all agent configuration files
    shutil.copytree(backup_path, config_path, dirs_exist_ok=True)
    
    # Validate each agent configuration
    for config_file in os.listdir(config_path):
        if config_file.endswith('.json'):
            with open(os.path.join(config_path, config_file)) as f:
                try:
                    config = json.load(f)
                    print(f"✅ {config_file}: Valid")
                except json.JSONDecodeError as e:
                    print(f"❌ {config_file}: Invalid - {e}")
```

---

## 4. Testing and Validation

### 4.1 Recovery Testing Schedule

#### Monthly Tests
- **Database backup/restore test**
- **Configuration recovery test**
- **Single agent failure recovery**

#### Quarterly Tests
- **Complete system recovery test**
- **Failover procedure test**
- **Cross-platform recovery test**

#### Annual Tests
- **Full disaster recovery simulation**
- **Multi-site failover test**
- **Regulatory compliance validation**

### 4.2 Test Procedures

#### Database Recovery Test
```bash
#!/bin/bash
# Monthly database recovery test

# Create test backup
sqlite3 data/quantai.db ".backup data/test_backup.db"

# Simulate corruption
echo "CORRUPT DATA" >> data/quantai.db

# Attempt recovery
cp data/test_backup.db data/quantai_recovered.db

# Validate recovery
python scripts/validate_database.py data/quantai_recovered.db

# Restore original
mv data/quantai_recovered.db data/quantai.db
rm data/test_backup.db
```

#### System Recovery Test
```python
# Quarterly system recovery test
import subprocess
import time

def test_system_recovery():
    # Stop system
    subprocess.run(["docker-compose", "down"])
    
    # Simulate data loss
    subprocess.run(["mv", "data/quantai.db", "data/quantai.db.backup"])
    
    # Start recovery
    start_time = time.time()
    subprocess.run(["python", "scripts/disaster_recovery.py"])
    
    # Measure recovery time
    recovery_time = time.time() - start_time
    
    # Validate system
    result = subprocess.run(["python", "scripts/system_health_check.py"])
    
    # Restore backup
    subprocess.run(["mv", "data/quantai.db.backup", "data/quantai.db"])
    
    return {
        'recovery_time': recovery_time,
        'success': result.returncode == 0
    }
```

---

## 5. Communication Plan

### 5.1 Stakeholder Notification

#### Internal Stakeholders
- **CEO/CTO:** Immediate notification for Severity 1 incidents
- **Operations Team:** All incidents
- **Development Team:** Technical incidents
- **Compliance Officer:** Regulatory-related incidents

#### External Stakeholders
- **Clients:** Service disruption notifications
- **Regulators:** Compliance-related incidents
- **Service Providers:** Infrastructure-related issues

### 5.2 Communication Templates

#### Incident Notification
```
SUBJECT: [URGENT] QuantAI System Incident - Severity [X]

Incident ID: [ID]
Time: [TIMESTAMP]
Severity: [1/2/3]
Status: [INVESTIGATING/RESOLVING/RESOLVED]

Description: [Brief description of the incident]

Impact: [Description of business impact]

Actions Taken: [Current response actions]

Next Update: [Time of next update]

Contact: [Emergency contact information]
```

#### Recovery Status Update
```
SUBJECT: QuantAI Recovery Update - [STATUS]

Incident ID: [ID]
Recovery Progress: [X]%
Estimated Completion: [TIME]

Current Status: [Detailed status update]

Completed Actions:
- [Action 1]
- [Action 2]

Remaining Actions:
- [Action 1]
- [Action 2]

Next Update: [Time]
```

---

## 6. Post-Incident Procedures

### 6.1 Post-Incident Review

#### Review Meeting (Within 48 hours)
- **Incident timeline reconstruction**
- **Root cause analysis**
- **Response effectiveness evaluation**
- **Recovery time assessment**
- **Communication effectiveness review**

#### Documentation Requirements
- **Incident report**
- **Timeline of events**
- **Root cause analysis**
- **Lessons learned**
- **Action items for improvement**

### 6.2 Continuous Improvement

#### Process Updates
- **Update recovery procedures based on lessons learned**
- **Improve monitoring and alerting**
- **Enhance backup strategies**
- **Update communication plans**

#### Training Updates
- **Conduct additional training if needed**
- **Update emergency contact lists**
- **Review and update documentation**
- **Schedule additional testing if required**

---

## 7. Emergency Contacts

### Internal Contacts
- **Primary On-Call:** [Name] - [Phone] - [Email]
- **Secondary On-Call:** [Name] - [Phone] - [Email]
- **Technical Lead:** [Name] - [Phone] - [Email]
- **Operations Manager:** [Name] - [Phone] - [Email]

### External Contacts
- **IBKR Support:** +1-877-442-2757
- **Alpaca Support:** support@alpaca.markets
- **AWS Support:** [Account-specific number]
- **Cloud Provider:** [Support contact]

### Regulatory Contacts
- **SEC:** [Contact information]
- **FINRA:** [Contact information]
- **Legal Counsel:** [Contact information]

---

*This document is reviewed and updated quarterly. Last review: 2025-01-04*
