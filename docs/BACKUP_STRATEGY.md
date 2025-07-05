# QuantAI Backup Strategy

**Document Version:** 1.0  
**Last Updated:** 2025-01-04  
**Review Schedule:** Quarterly

---

## Overview

This document outlines the comprehensive backup strategy for the QuantAI quantitative trading system, ensuring data protection, regulatory compliance, and business continuity.

**Key Objectives:**
- **Data Protection:** Prevent data loss from system failures, corruption, or human error
- **Regulatory Compliance:** Meet 7-year audit trail retention requirements
- **Business Continuity:** Enable rapid recovery with minimal downtime
- **Cost Optimization:** Balance protection needs with storage costs

---

## 1. Data Classification and Backup Requirements

### 1.1 Critical Data (Tier 1) - Real-time Backup
**Data Types:**
- Trading positions and orders
- Account balances and P&L statements
- Risk management settings and limits
- Compliance audit trails
- System configuration files
- API keys and security credentials (encrypted)

**Backup Requirements:**
- **Frequency:** Real-time replication + 15-minute snapshots
- **Retention:** 7 years (regulatory requirement)
- **Recovery Time:** < 15 minutes
- **Storage:** 3 copies (local, cloud, offsite)

### 1.2 Important Data (Tier 2) - Hourly Backup
**Data Types:**
- Historical market data
- Strategy performance metrics
- System logs and monitoring data
- User preferences and settings
- Backtesting results

**Backup Requirements:**
- **Frequency:** Hourly snapshots
- **Retention:** 2 years
- **Recovery Time:** < 1 hour
- **Storage:** 2 copies (local, cloud)

### 1.3 Standard Data (Tier 3) - Daily Backup
**Data Types:**
- Cached market data
- Temporary files
- Development artifacts
- Non-critical logs

**Backup Requirements:**
- **Frequency:** Daily snapshots
- **Retention:** 30 days
- **Recovery Time:** < 4 hours
- **Storage:** 1 copy (local or cloud)

---

## 2. Backup Infrastructure

### 2.1 Three-Tier Backup Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BACKUP ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│ TIER 1: LOCAL     │ NAS/SAN Storage │ Real-time + Hourly   │
├─────────────────────────────────────────────────────────────┤
│ TIER 2: CLOUD     │ AWS S3/Azure    │ Daily + Weekly       │
├─────────────────────────────────────────────────────────────┤
│ TIER 3: OFFSITE   │ Physical Media  │ Weekly + Monthly     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Local Backup System (Tier 1)
**Technology:** Network Attached Storage (NAS) with RAID 6
**Capacity:** 10TB usable storage
**Features:**
- Automatic snapshots every 15 minutes
- Deduplication and compression
- Automated integrity checks
- Hot-swappable drives

**Configuration:**
```bash
# NAS backup configuration
BACKUP_LOCAL_PATH="/nas/quantai-backups"
SNAPSHOT_INTERVAL="15m"
RETENTION_POLICY="7y"
COMPRESSION="lz4"
DEDUPLICATION="enabled"
```

### 2.3 Cloud Backup System (Tier 2)
**Technology:** AWS S3 with versioning and lifecycle policies
**Capacity:** Unlimited with intelligent tiering
**Features:**
- Cross-region replication
- Encryption at rest and in transit
- Automated lifecycle management
- Cost optimization with storage classes

**Configuration:**
```json
{
  "backup_bucket": "quantai-backups-primary",
  "replication_bucket": "quantai-backups-replica",
  "encryption": "AES-256",
  "versioning": "enabled",
  "lifecycle_policy": {
    "transition_to_ia": "30_days",
    "transition_to_glacier": "90_days",
    "transition_to_deep_archive": "365_days"
  }
}
```

### 2.4 Offsite Backup System (Tier 3)
**Technology:** Encrypted portable drives stored in secure facility
**Capacity:** 5TB per drive, rotated monthly
**Features:**
- Hardware encryption (AES-256)
- Fireproof and waterproof storage
- Geographic separation (>100 miles)
- Monthly rotation schedule

---

## 3. Backup Procedures

### 3.1 Automated Backup Scripts

#### Real-time Database Backup
```python
#!/usr/bin/env python3
"""Real-time database backup using SQLite WAL mode."""

import sqlite3
import shutil
import time
import hashlib
from datetime import datetime
from pathlib import Path

class RealTimeBackup:
    def __init__(self, db_path, backup_dir):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_snapshot(self):
        """Create a point-in-time snapshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"quantai_{timestamp}.db"
        
        # Use SQLite backup API for consistent snapshot
        source_conn = sqlite3.connect(str(self.db_path))
        backup_conn = sqlite3.connect(str(backup_path))
        
        source_conn.backup(backup_conn)
        
        source_conn.close()
        backup_conn.close()
        
        # Verify backup integrity
        if self.verify_backup(backup_path):
            print(f"✅ Backup created: {backup_path}")
            return backup_path
        else:
            backup_path.unlink()  # Delete corrupted backup
            raise Exception("Backup verification failed")
    
    def verify_backup(self, backup_path):
        """Verify backup integrity."""
        try:
            conn = sqlite3.connect(str(backup_path))
            result = conn.execute("PRAGMA integrity_check").fetchone()
            conn.close()
            return result[0] == "ok"
        except Exception:
            return False

# Usage
backup_system = RealTimeBackup("data/quantai.db", "/nas/quantai-backups/realtime")
backup_system.create_snapshot()
```

#### Incremental Backup System
```bash
#!/bin/bash
# Incremental backup script using rsync

BACKUP_SOURCE="/opt/quantai"
BACKUP_DEST="/nas/quantai-backups/incremental"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CURRENT_BACKUP="$BACKUP_DEST/current"
BACKUP_LOG="/var/log/quantai-backup.log"

# Create backup directory
mkdir -p "$BACKUP_DEST/$TIMESTAMP"

# Perform incremental backup
rsync -av --delete \
    --link-dest="$CURRENT_BACKUP" \
    --exclude="*.tmp" \
    --exclude="*.log" \
    --exclude="cache/" \
    "$BACKUP_SOURCE/" \
    "$BACKUP_DEST/$TIMESTAMP/" \
    >> "$BACKUP_LOG" 2>&1

# Update current symlink
rm -f "$CURRENT_BACKUP"
ln -s "$TIMESTAMP" "$CURRENT_BACKUP"

# Cleanup old backups (keep 30 days)
find "$BACKUP_DEST" -maxdepth 1 -type d -name "20*" -mtime +30 -exec rm -rf {} \;

echo "$(date): Incremental backup completed - $TIMESTAMP" >> "$BACKUP_LOG"
```

### 3.2 Cloud Backup Automation

#### AWS S3 Backup Script
```python
#!/usr/bin/env python3
"""Automated cloud backup to AWS S3."""

import boto3
import os
import gzip
import json
from datetime import datetime
from pathlib import Path
from cryptography.fernet import Fernet

class CloudBackup:
    def __init__(self, bucket_name, encryption_key):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.cipher = Fernet(encryption_key)
    
    def backup_file(self, local_path, s3_key):
        """Backup a file to S3 with encryption and compression."""
        
        # Read and compress file
        with open(local_path, 'rb') as f:
            data = f.read()
        
        compressed_data = gzip.compress(data)
        encrypted_data = self.cipher.encrypt(compressed_data)
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=encrypted_data,
            ServerSideEncryption='AES256',
            Metadata={
                'original_size': str(len(data)),
                'compressed_size': str(len(compressed_data)),
                'backup_timestamp': datetime.now().isoformat()
            }
        )
        
        print(f"✅ Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
    
    def backup_directory(self, local_dir, s3_prefix):
        """Backup entire directory to S3."""
        local_path = Path(local_dir)
        
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative_path}"
                self.backup_file(str(file_path), s3_key)

# Usage
encryption_key = os.environ.get('BACKUP_ENCRYPTION_KEY')
backup_system = CloudBackup('quantai-backups', encryption_key)

# Daily backup
timestamp = datetime.now().strftime("%Y%m%d")
backup_system.backup_directory('/opt/quantai/data', f'daily/{timestamp}/data')
backup_system.backup_directory('/opt/quantai/config', f'daily/{timestamp}/config')
```

### 3.3 Backup Scheduling

#### Cron Configuration
```bash
# /etc/crontab - QuantAI Backup Schedule

# Real-time snapshots (every 15 minutes)
*/15 * * * * quantai /opt/quantai/scripts/realtime_backup.py

# Hourly incremental backup
0 * * * * quantai /opt/quantai/scripts/incremental_backup.sh

# Daily cloud backup (2 AM)
0 2 * * * quantai /opt/quantai/scripts/cloud_backup.py

# Weekly full backup (Sunday 1 AM)
0 1 * * 0 quantai /opt/quantai/scripts/full_backup.sh

# Monthly offsite backup (1st of month, 3 AM)
0 3 1 * * quantai /opt/quantai/scripts/offsite_backup.sh

# Backup verification (daily at 6 AM)
0 6 * * * quantai /opt/quantai/scripts/verify_backups.py
```

---

## 4. Backup Verification and Testing

### 4.1 Automated Verification

#### Backup Integrity Checker
```python
#!/usr/bin/env python3
"""Automated backup verification system."""

import sqlite3
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta

class BackupVerifier:
    def __init__(self, backup_dir):
        self.backup_dir = Path(backup_dir)
        self.verification_log = []
    
    def verify_database_backup(self, backup_path):
        """Verify database backup integrity."""
        try:
            conn = sqlite3.connect(str(backup_path))
            
            # Check database integrity
            integrity_result = conn.execute("PRAGMA integrity_check").fetchone()
            if integrity_result[0] != "ok":
                return False, f"Integrity check failed: {integrity_result[0]}"
            
            # Check critical tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            
            required_tables = ['trades', 'positions', 'accounts', 'audit_trail']
            existing_tables = [table[0] for table in tables]
            
            missing_tables = set(required_tables) - set(existing_tables)
            if missing_tables:
                return False, f"Missing tables: {missing_tables}"
            
            # Check data consistency
            trade_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            position_count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
            
            conn.close()
            
            return True, f"Database verified: {trade_count} trades, {position_count} positions"
            
        except Exception as e:
            return False, f"Verification error: {str(e)}"
    
    def verify_file_backup(self, backup_path, original_path):
        """Verify file backup by comparing checksums."""
        try:
            if not backup_path.exists():
                return False, "Backup file does not exist"
            
            if not original_path.exists():
                return True, "Original file missing, backup exists"
            
            # Compare file sizes
            if backup_path.stat().st_size != original_path.stat().st_size:
                return False, "File sizes do not match"
            
            # Compare checksums
            backup_hash = self.calculate_checksum(backup_path)
            original_hash = self.calculate_checksum(original_path)
            
            if backup_hash != original_hash:
                return False, "Checksums do not match"
            
            return True, "File backup verified"
            
        except Exception as e:
            return False, f"Verification error: {str(e)}"
    
    def calculate_checksum(self, file_path):
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def run_verification(self):
        """Run comprehensive backup verification."""
        verification_results = {
            'timestamp': datetime.now().isoformat(),
            'results': []
        }
        
        # Verify recent database backups
        db_backups = sorted(self.backup_dir.glob("quantai_*.db"))[-5:]  # Last 5 backups
        
        for backup_path in db_backups:
            success, message = self.verify_database_backup(backup_path)
            verification_results['results'].append({
                'type': 'database',
                'file': str(backup_path),
                'success': success,
                'message': message
            })
        
        # Save verification results
        results_file = self.backup_dir / f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(verification_results, f, indent=2)
        
        return verification_results

# Usage
verifier = BackupVerifier('/nas/quantai-backups')
results = verifier.run_verification()
print(f"Verification completed: {len(results['results'])} backups checked")
```

### 4.2 Recovery Testing

#### Monthly Recovery Test
```bash
#!/bin/bash
# Monthly backup recovery test

TEST_DIR="/tmp/quantai-recovery-test"
BACKUP_DIR="/nas/quantai-backups"
LOG_FILE="/var/log/quantai-recovery-test.log"

echo "$(date): Starting recovery test" >> "$LOG_FILE"

# Create test environment
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Find latest backup
LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/quantai_*.db | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "$(date): ERROR - No backup found" >> "$LOG_FILE"
    exit 1
fi

# Copy backup to test directory
cp "$LATEST_BACKUP" "$TEST_DIR/quantai_test.db"

# Test database recovery
python3 << EOF
import sqlite3
import sys

try:
    conn = sqlite3.connect('quantai_test.db')
    
    # Test basic queries
    trade_count = conn.execute('SELECT COUNT(*) FROM trades').fetchone()[0]
    position_count = conn.execute('SELECT COUNT(*) FROM positions').fetchone()[0]
    
    print(f"Recovery test successful: {trade_count} trades, {position_count} positions")
    
    conn.close()
    sys.exit(0)
    
except Exception as e:
    print(f"Recovery test failed: {e}")
    sys.exit(1)
EOF

RECOVERY_RESULT=$?

if [ $RECOVERY_RESULT -eq 0 ]; then
    echo "$(date): Recovery test PASSED" >> "$LOG_FILE"
else
    echo "$(date): Recovery test FAILED" >> "$LOG_FILE"
fi

# Cleanup
rm -rf "$TEST_DIR"

exit $RECOVERY_RESULT
```

---

## 5. Backup Monitoring and Alerting

### 5.1 Backup Health Monitoring

#### Backup Status Dashboard
```python
#!/usr/bin/env python3
"""Backup status monitoring dashboard."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

class BackupMonitor:
    def __init__(self, backup_dir):
        self.backup_dir = Path(backup_dir)
    
    def get_backup_status(self):
        """Get comprehensive backup status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'unknown',
            'backup_counts': {},
            'latest_backups': {},
            'alerts': []
        }
        
        # Check backup counts by age
        now = datetime.now()
        time_ranges = {
            'last_hour': timedelta(hours=1),
            'last_day': timedelta(days=1),
            'last_week': timedelta(weeks=1),
            'last_month': timedelta(days=30)
        }
        
        for range_name, time_delta in time_ranges.items():
            cutoff_time = now - time_delta
            backup_files = [
                f for f in self.backup_dir.glob("quantai_*.db")
                if datetime.fromtimestamp(f.stat().st_mtime) > cutoff_time
            ]
            status['backup_counts'][range_name] = len(backup_files)
        
        # Find latest backups
        latest_backups = sorted(
            self.backup_dir.glob("quantai_*.db"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )[:5]
        
        for backup in latest_backups:
            backup_time = datetime.fromtimestamp(backup.stat().st_mtime)
            status['latest_backups'][str(backup)] = {
                'timestamp': backup_time.isoformat(),
                'size_mb': round(backup.stat().st_size / 1024 / 1024, 2),
                'age_hours': round((now - backup_time).total_seconds() / 3600, 1)
            }
        
        # Generate alerts
        if status['backup_counts']['last_hour'] == 0:
            status['alerts'].append({
                'severity': 'critical',
                'message': 'No backups created in the last hour'
            })
        
        if status['backup_counts']['last_day'] < 24:
            status['alerts'].append({
                'severity': 'warning',
                'message': f"Only {status['backup_counts']['last_day']} backups in last 24 hours"
            })
        
        # Determine overall health
        if not status['alerts']:
            status['overall_health'] = 'healthy'
        elif any(alert['severity'] == 'critical' for alert in status['alerts']):
            status['overall_health'] = 'critical'
        else:
            status['overall_health'] = 'warning'
        
        return status
    
    def send_alerts(self, status):
        """Send alerts for backup issues."""
        critical_alerts = [
            alert for alert in status['alerts']
            if alert['severity'] == 'critical'
        ]
        
        if critical_alerts:
            # Send email/Slack notification
            alert_message = f"""
            CRITICAL BACKUP ALERT
            
            Time: {status['timestamp']}
            Issues: {len(critical_alerts)}
            
            Details:
            {chr(10).join(alert['message'] for alert in critical_alerts)}
            
            Please investigate immediately.
            """
            
            # Implementation would send actual alerts
            print("🚨 CRITICAL BACKUP ALERT SENT")
            print(alert_message)

# Usage
monitor = BackupMonitor('/nas/quantai-backups')
status = monitor.get_backup_status()
monitor.send_alerts(status)

print(f"Backup Health: {status['overall_health']}")
print(f"Recent backups: {status['backup_counts']['last_day']}")
```

### 5.2 Automated Alerting

#### Backup Alert Configuration
```json
{
  "alert_rules": [
    {
      "name": "missing_hourly_backup",
      "condition": "no_backup_in_last_hour",
      "severity": "critical",
      "notification_channels": ["email", "slack", "sms"]
    },
    {
      "name": "backup_verification_failed",
      "condition": "verification_failure",
      "severity": "critical",
      "notification_channels": ["email", "slack"]
    },
    {
      "name": "low_backup_frequency",
      "condition": "less_than_20_backups_per_day",
      "severity": "warning",
      "notification_channels": ["email"]
    },
    {
      "name": "backup_size_anomaly",
      "condition": "backup_size_change_over_50_percent",
      "severity": "warning",
      "notification_channels": ["email"]
    }
  ],
  "notification_settings": {
    "email": {
      "smtp_server": "smtp.gmail.com",
      "smtp_port": 587,
      "recipients": ["admin@quantai.com", "ops@quantai.com"]
    },
    "slack": {
      "webhook_url": "https://hooks.slack.com/services/...",
      "channel": "#quantai-alerts"
    },
    "sms": {
      "provider": "twilio",
      "recipients": ["+1234567890"]
    }
  }
}
```

---

## 6. Compliance and Retention

### 6.1 Regulatory Requirements

#### Financial Industry Compliance
- **SEC Rule 17a-4:** Electronic record retention requirements
- **FINRA Rule 4511:** Books and records requirements
- **SOX Compliance:** Internal controls and audit trails

#### Retention Schedule
```json
{
  "retention_policies": {
    "audit_trails": {
      "retention_period": "7_years",
      "regulatory_basis": "SEC_17a-4",
      "storage_requirements": ["immutable", "encrypted", "geographically_distributed"]
    },
    "trade_records": {
      "retention_period": "7_years",
      "regulatory_basis": "FINRA_4511",
      "storage_requirements": ["tamper_evident", "readily_accessible"]
    },
    "financial_statements": {
      "retention_period": "7_years",
      "regulatory_basis": "SOX_404",
      "storage_requirements": ["secure", "backed_up", "auditable"]
    },
    "system_logs": {
      "retention_period": "2_years",
      "regulatory_basis": "internal_policy",
      "storage_requirements": ["compressed", "searchable"]
    }
  }
}
```

### 6.2 Compliance Monitoring

#### Retention Compliance Checker
```python
#!/usr/bin/env python3
"""Monitor backup retention compliance."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

class ComplianceMonitor:
    def __init__(self, backup_dir):
        self.backup_dir = Path(backup_dir)
        self.retention_policies = {
            'audit_trails': timedelta(days=7*365),  # 7 years
            'trade_records': timedelta(days=7*365),  # 7 years
            'system_logs': timedelta(days=2*365),   # 2 years
            'temp_files': timedelta(days=30)        # 30 days
        }
    
    def check_retention_compliance(self):
        """Check if backups meet retention requirements."""
        compliance_report = {
            'timestamp': datetime.now().isoformat(),
            'compliant': True,
            'violations': [],
            'recommendations': []
        }
        
        # Check for oldest required backups
        now = datetime.now()
        
        for data_type, retention_period in self.retention_policies.items():
            oldest_required = now - retention_period
            
            # Find backups for this data type
            pattern = f"*{data_type}*.db" if data_type != 'temp_files' else "temp_*"
            backups = list(self.backup_dir.glob(pattern))
            
            if not backups:
                compliance_report['violations'].append({
                    'type': 'missing_backups',
                    'data_type': data_type,
                    'message': f'No backups found for {data_type}'
                })
                compliance_report['compliant'] = False
                continue
            
            # Check oldest backup
            oldest_backup = min(backups, key=lambda f: f.stat().st_mtime)
            oldest_backup_time = datetime.fromtimestamp(oldest_backup.stat().st_mtime)
            
            if oldest_backup_time > oldest_required:
                compliance_report['violations'].append({
                    'type': 'insufficient_retention',
                    'data_type': data_type,
                    'oldest_backup': oldest_backup_time.isoformat(),
                    'required_date': oldest_required.isoformat(),
                    'message': f'Oldest {data_type} backup is too recent for compliance'
                })
                compliance_report['compliant'] = False
        
        return compliance_report

# Usage
monitor = ComplianceMonitor('/nas/quantai-backups')
report = monitor.check_retention_compliance()

if report['compliant']:
    print("✅ All retention policies are compliant")
else:
    print(f"❌ {len(report['violations'])} compliance violations found")
    for violation in report['violations']:
        print(f"   - {violation['message']}")
```

---

*This backup strategy is reviewed and updated quarterly to ensure continued effectiveness and compliance.*
