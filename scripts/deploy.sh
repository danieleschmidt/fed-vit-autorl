#!/bin/bash
# Fed-ViT-AutoRL Production Deployment Script

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_ROOT}/.env"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"

# Default values
ENVIRONMENT="production"
PROFILE="production"
SCALE_CLIENTS=5
BACKUP_ENABLED=true
MONITORING_ENABLED=true

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

# Help function
show_help() {
    cat << EOF
Fed-ViT-AutoRL Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    deploy      Deploy the application
    stop        Stop all services
    restart     Restart all services
    status      Show service status
    logs        Show service logs
    backup      Create backup
    restore     Restore from backup
    update      Update application
    scale       Scale services

Options:
    -e, --env ENV           Environment (production, staging, development)
    -p, --profile PROFILE   Docker compose profile
    -s, --scale N           Number of client replicas
    -h, --help              Show this help

Examples:
    $0 deploy
    $0 -e production -s 10 deploy
    $0 logs fed-server
    $0 scale fed-client=8
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -p|--profile)
                PROFILE="$2"
                shift 2
                ;;
            -s|--scale)
                SCALE_CLIENTS="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            deploy|stop|restart|status|logs|backup|restore|update|scale)
                COMMAND="$1"
                shift
                break
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Store remaining arguments
    ARGS=("$@")
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df "${PROJECT_ROOT}" | tail -1 | awk '{print $4}')
    if [[ $available_space -lt 10485760 ]]; then  # 10GB in KB
        log_warning "Less than 10GB disk space available"
    fi
    
    log_success "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment for: $ENVIRONMENT"
    
    # Create .env file if it doesn't exist
    if [[ ! -f "$ENV_FILE" ]]; then
        log_info "Creating environment file..."
        cat > "$ENV_FILE" << EOF
# Fed-ViT-AutoRL Environment Configuration
COMPOSE_PROJECT_NAME=fed-vit-autorl
ENVIRONMENT=$ENVIRONMENT
POSTGRES_PASSWORD=$(openssl rand -base64 32)
GRAFANA_SECRET_KEY=$(openssl rand -base64 32)
JUPYTER_TOKEN=$(openssl rand -base64 16)
EOF
        log_success "Environment file created"
    fi
    
    # Create necessary directories
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/data"
    mkdir -p "${PROJECT_ROOT}/models"
    mkdir -p "${PROJECT_ROOT}/backups"
    mkdir -p "${PROJECT_ROOT}/certs"
    
    # Set proper permissions
    chmod 755 "${PROJECT_ROOT}/logs"
    chmod 755 "${PROJECT_ROOT}/data"
    chmod 755 "${PROJECT_ROOT}/models"
    chmod 700 "${PROJECT_ROOT}/backups"
    chmod 700 "${PROJECT_ROOT}/certs"
    
    log_success "Environment setup completed"
}

# Generate SSL certificates (self-signed for development)
generate_certificates() {
    local cert_dir="${PROJECT_ROOT}/certs"
    
    if [[ ! -f "${cert_dir}/server.crt" ]]; then
        log_info "Generating SSL certificates..."
        
        openssl req -new -newkey rsa:4096 -days 365 -nodes -x509 \
            -subj "/C=US/ST=CA/L=SF/O=Fed-ViT-AutoRL/CN=localhost" \
            -keyout "${cert_dir}/server.key" \
            -out "${cert_dir}/server.crt"
        
        chmod 600 "${cert_dir}/server.key"
        chmod 644 "${cert_dir}/server.crt"
        
        log_success "SSL certificates generated"
    fi
}

# Deploy application
deploy_application() {
    log_info "Deploying Fed-ViT-AutoRL..."
    
    # Pull latest images
    log_info "Pulling latest images..."
    docker compose --profile "$PROFILE" pull
    
    # Build custom images
    log_info "Building application images..."
    docker compose build --no-cache
    
    # Start services
    log_info "Starting services..."
    docker compose --profile "$PROFILE" up -d \
        --scale fed-client="$SCALE_CLIENTS"
    
    # Wait for services to be ready
    wait_for_services
    
    log_success "Deployment completed successfully"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    local max_attempts=60
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if docker compose exec -T fed-server python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" &> /dev/null; then
            log_success "Fed-ViT server is ready"
            break
        fi
        
        attempt=$((attempt + 1))
        sleep 5
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Services failed to start within expected time"
            exit 1
        fi
    done
}

# Stop services
stop_services() {
    log_info "Stopping Fed-ViT-AutoRL services..."
    docker compose --profile "$PROFILE" down
    log_success "Services stopped"
}

# Restart services
restart_services() {
    log_info "Restarting Fed-ViT-AutoRL services..."
    docker compose --profile "$PROFILE" restart
    log_success "Services restarted"
}

# Show service status
show_status() {
    log_info "Fed-ViT-AutoRL Service Status:"
    docker compose --profile "$PROFILE" ps
    
    echo
    log_info "Container Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\\t{{.CPUPerc}}\\t{{.MemUsage}}\\t{{.NetIO}}"
}

# Show logs
show_logs() {
    local service="${ARGS[0]:-}"
    
    if [[ -n "$service" ]]; then
        docker compose logs -f "$service"
    else
        docker compose logs -f
    fi
}

# Create backup
create_backup() {
    log_info "Creating backup..."
    
    local backup_dir="${PROJECT_ROOT}/backups"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="fedvit_backup_${timestamp}.tar.gz"
    
    # Create backup
    tar -czf "${backup_dir}/${backup_name}" \
        -C "${PROJECT_ROOT}" \
        configs/ data/ models/ logs/ docker-compose.yml .env
    
    log_success "Backup created: ${backup_name}"
}

# Scale services
scale_services() {
    local scale_arg="${ARGS[0]:-}"
    
    if [[ -z "$scale_arg" ]]; then
        log_error "Scale argument required (e.g., fed-client=8)"
        exit 1
    fi
    
    log_info "Scaling services: $scale_arg"
    docker compose --profile "$PROFILE" up -d --scale "$scale_arg" --no-recreate
    log_success "Services scaled successfully"
}

# Update application
update_application() {
    log_info "Updating Fed-ViT-AutoRL..."
    
    # Create backup before update
    create_backup
    
    # Pull latest changes (if in git repo)
    if [[ -d "${PROJECT_ROOT}/.git" ]]; then
        log_info "Pulling latest changes..."
        git -C "$PROJECT_ROOT" pull
    fi
    
    # Rebuild and deploy
    deploy_application
    
    log_success "Update completed"
}

# Health check
health_check() {
    log_info "Running health checks..."
    
    local failed=0
    
    # Check server health
    if ! docker compose exec -T fed-server python -c "import requests; r = requests.get('http://localhost:8000/health', timeout=5); exit(0 if r.status_code == 200 else 1)" 2>/dev/null; then
        log_error "Fed-ViT server health check failed"
        failed=1
    else
        log_success "Fed-ViT server is healthy"
    fi
    
    # Check database
    if docker compose exec -T postgres pg_isready -U fedvit &> /dev/null; then
        log_success "Database is healthy"
    else
        log_error "Database health check failed"
        failed=1
    fi
    
    # Check Redis
    if docker compose exec -T redis redis-cli ping | grep -q PONG; then
        log_success "Redis is healthy"
    else
        log_error "Redis health check failed"
        failed=1
    fi
    
    if [[ $failed -eq 0 ]]; then
        log_success "All health checks passed"
    else
        log_error "Some health checks failed"
        exit 1
    fi
}

# Cleanup old resources
cleanup() {
    log_info "Cleaning up old resources..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful in production)
    if [[ "$ENVIRONMENT" != "production" ]]; then
        docker volume prune -f
    fi
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Parse arguments
    if [[ $# -eq 0 ]]; then
        show_help
        exit 1
    fi
    
    parse_args "$@"
    
    # Check if command is set
    if [[ -z "${COMMAND:-}" ]]; then
        log_error "No command specified"
        show_help
        exit 1
    fi
    
    # Execute command
    case "$COMMAND" in
        deploy)
            check_prerequisites
            setup_environment
            generate_certificates
            deploy_application
            health_check
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        backup)
            create_backup
            ;;
        update)
            update_application
            ;;
        scale)
            scale_services
            ;;
        health)
            health_check
            ;;
        cleanup)
            cleanup
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"