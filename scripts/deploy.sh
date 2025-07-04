#!/bin/bash

# QuantAI AutoGen Deployment Script
# This script handles deployment to different environments

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-development}"
VERSION="${2:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_warning ".env file not found, copying from .env.example"
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        log_warning "Please edit .env file with your configuration"
    fi
    
    log_success "Prerequisites check completed"
}

# Setup directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p "$PROJECT_ROOT/data/memory"
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/config"
    mkdir -p "$PROJECT_ROOT/monitoring/grafana/dashboards"
    mkdir -p "$PROJECT_ROOT/monitoring/grafana/datasources"
    mkdir -p "$PROJECT_ROOT/nginx/ssl"
    
    log_success "Directories created"
}

# Generate SSL certificates for development
generate_ssl_certs() {
    if [ "$ENVIRONMENT" = "development" ]; then
        log_info "Generating self-signed SSL certificates for development..."
        
        SSL_DIR="$PROJECT_ROOT/nginx/ssl"
        
        if [ ! -f "$SSL_DIR/cert.pem" ]; then
            openssl req -x509 -newkey rsa:4096 -keyout "$SSL_DIR/key.pem" \
                -out "$SSL_DIR/cert.pem" -days 365 -nodes \
                -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
            
            log_success "SSL certificates generated"
        else
            log_info "SSL certificates already exist"
        fi
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main application
    docker build -t quantai-app:$VERSION .
    
    # Build dashboard (if Dockerfile.dashboard exists)
    if [ -f "Dockerfile.dashboard" ]; then
        docker build -f Dockerfile.dashboard -t quantai-dashboard:$VERSION .
    fi
    
    log_success "Docker images built"
}

# Deploy to development environment
deploy_development() {
    log_info "Deploying to development environment..."
    
    cd "$PROJECT_ROOT"
    
    # Stop existing containers
    docker-compose down
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_service_health
    
    log_success "Development deployment completed"
}

# Deploy to staging environment
deploy_staging() {
    log_info "Deploying to staging environment..."
    
    cd "$PROJECT_ROOT"
    
    # Use staging compose file if it exists
    COMPOSE_FILE="docker-compose.staging.yml"
    if [ -f "$COMPOSE_FILE" ]; then
        docker-compose -f "$COMPOSE_FILE" down
        docker-compose -f "$COMPOSE_FILE" up -d
    else
        log_warning "Staging compose file not found, using default"
        docker-compose down
        docker-compose up -d
    fi
    
    log_success "Staging deployment completed"
}

# Deploy to production environment
deploy_production() {
    log_info "Deploying to production environment..."
    
    # Production deployment would typically use Kubernetes or similar
    log_warning "Production deployment requires manual configuration"
    log_info "Please refer to docs/deployment.md for production setup"
    
    # Basic production checks
    if [ -f "$PROJECT_ROOT/.env" ]; then
        if grep -q "QUANTAI_ENV=production" "$PROJECT_ROOT/.env"; then
            log_success "Production environment configured"
        else
            log_error "Environment not set to production in .env file"
            exit 1
        fi
    fi
}

# Check service health
check_service_health() {
    log_info "Checking service health..."
    
    # Check main application
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Main application is healthy"
    else
        log_warning "Main application health check failed"
    fi
    
    # Check dashboard
    if curl -f http://localhost:8501 &> /dev/null; then
        log_success "Dashboard is healthy"
    else
        log_warning "Dashboard health check failed"
    fi
    
    # Check database
    if docker-compose exec -T postgres pg_isready -U quantai &> /dev/null; then
        log_success "Database is healthy"
    else
        log_warning "Database health check failed"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping &> /dev/null; then
        log_success "Redis is healthy"
    else
        log_warning "Redis health check failed"
    fi
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # This would run actual migrations in a real system
    docker-compose exec quantai-app python -c "
from quantai.core.database import init_database
init_database()
print('Database initialized')
"
    
    log_success "Database migrations completed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create Prometheus configuration
    cat > "$PROJECT_ROOT/monitoring/prometheus.yml" << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'quantai-app'
    static_configs:
      - targets: ['quantai-app:8080']
  
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
    
    # Create Grafana datasource configuration
    mkdir -p "$PROJECT_ROOT/monitoring/grafana/datasources"
    cat > "$PROJECT_ROOT/monitoring/grafana/datasources/prometheus.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    log_success "Monitoring setup completed"
}

# Main deployment function
main() {
    log_info "Starting QuantAI AutoGen deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    
    check_prerequisites
    setup_directories
    generate_ssl_certs
    setup_monitoring
    build_images
    
    case $ENVIRONMENT in
        "development")
            deploy_development
            run_migrations
            ;;
        "staging")
            deploy_staging
            run_migrations
            ;;
        "production")
            deploy_production
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            log_info "Supported environments: development, staging, production"
            exit 1
            ;;
    esac
    
    log_success "Deployment completed successfully!"
    
    # Show access information
    echo ""
    log_info "Access Information:"
    echo "  Main Application: http://localhost:8000"
    echo "  Dashboard: http://localhost:8501"
    echo "  Grafana: http://localhost:3000 (admin/admin)"
    echo "  Prometheus: http://localhost:9090"
    echo ""
    log_info "Check logs with: docker-compose logs -f"
}

# Show usage information
show_usage() {
    echo "Usage: $0 [environment] [version]"
    echo ""
    echo "Environments:"
    echo "  development (default) - Local development setup"
    echo "  staging              - Staging environment"
    echo "  production           - Production environment"
    echo ""
    echo "Examples:"
    echo "  $0                    # Deploy to development"
    echo "  $0 staging           # Deploy to staging"
    echo "  $0 production v1.0.0 # Deploy version v1.0.0 to production"
}

# Handle command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_usage
    exit 0
fi

# Run main function
main
