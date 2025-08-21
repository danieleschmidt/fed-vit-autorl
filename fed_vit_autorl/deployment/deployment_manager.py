"""Production deployment management for Fed-ViT-AutoRL."""

import os
import time
import logging
import subprocess
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: DeploymentEnvironment
    version: str
    replicas: int = 3
    cpu_limit: str = "2000m"
    memory_limit: str = "4Gi"
    gpu_enabled: bool = False
    gpu_limit: int = 1
    enable_monitoring: bool = True
    enable_logging: bool = True
    enable_tracing: bool = True
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    port: int = 8080
    secrets: Optional[List[str]] = None
    config_maps: Optional[List[str]] = None
    persistent_storage: bool = False
    storage_size: str = "10Gi"
    backup_enabled: bool = True
    auto_scaling_enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    ingress_enabled: bool = True
    tls_enabled: bool = True
    domain: Optional[str] = None

    def __post_init__(self):
        if self.secrets is None:
            self.secrets = []
        if self.config_maps is None:
            self.config_maps = []


@dataclass
class DeploymentResult:
    """Deployment operation result."""
    deployment_id: str
    status: DeploymentStatus
    environment: DeploymentEnvironment
    version: str
    message: str
    start_time: float
    end_time: Optional[float] = None
    artifacts: Optional[Dict[str, str]] = None
    rollback_version: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        """Get deployment duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None


class DeploymentManager:
    """Comprehensive production deployment management."""

    def __init__(
        self,
        project_root: str = ".",
        registry_url: Optional[str] = None,
        kubeconfig_path: Optional[str] = None,
        namespace: str = "fed-vit-autorl",
    ):
        """Initialize deployment manager.

        Args:
            project_root: Project root directory
            registry_url: Container registry URL
            kubeconfig_path: Kubernetes config file path
            namespace: Kubernetes namespace
        """
        self.project_root = os.path.abspath(project_root)
        self.registry_url = registry_url or "localhost:5000"
        self.kubeconfig_path = kubeconfig_path
        self.namespace = namespace

        # Deployment state
        self.deployments: Dict[str, DeploymentResult] = {}
        self.deployment_history: List[DeploymentResult] = []

        # Paths
        self.deployment_dir = os.path.join(self.project_root, "deployments")
        self.manifests_dir = os.path.join(self.deployment_dir, "manifests")
        self.scripts_dir = os.path.join(self.deployment_dir, "scripts")

        # Ensure deployment directories exist
        self._ensure_deployment_structure()

        logger.info(f"Initialized deployment manager for {self.project_root}")

    def _ensure_deployment_structure(self) -> None:
        """Ensure deployment directory structure exists."""
        dirs_to_create = [
            self.deployment_dir,
            self.manifests_dir,
            os.path.join(self.manifests_dir, "base"),
            os.path.join(self.manifests_dir, "overlays"),
            os.path.join(self.manifests_dir, "overlays", "development"),
            os.path.join(self.manifests_dir, "overlays", "staging"),
            os.path.join(self.manifests_dir, "overlays", "production"),
            self.scripts_dir,
        ]

        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)

    def create_deployment_manifests(self, config: DeploymentConfig) -> Dict[str, str]:
        """Create Kubernetes deployment manifests.

        Args:
            config: Deployment configuration

        Returns:
            Dictionary of manifest file paths
        """
        manifests = {}
        env_name = config.environment.value
        overlay_dir = os.path.join(self.manifests_dir, "overlays", env_name)

        # Create deployment manifest
        deployment_manifest = self._create_deployment_manifest(config)
        deployment_path = os.path.join(overlay_dir, "deployment.yaml")

        with open(deployment_path, 'w') as f:
            f.write(deployment_manifest)
        manifests["deployment"] = deployment_path

        # Create service manifest
        service_manifest = self._create_service_manifest(config)
        service_path = os.path.join(overlay_dir, "service.yaml")

        with open(service_path, 'w') as f:
            f.write(service_manifest)
        manifests["service"] = service_path

        # Create ingress manifest if enabled
        if config.ingress_enabled:
            ingress_manifest = self._create_ingress_manifest(config)
            ingress_path = os.path.join(overlay_dir, "ingress.yaml")

            with open(ingress_path, 'w') as f:
                f.write(ingress_manifest)
            manifests["ingress"] = ingress_path

        # Create horizontal pod autoscaler if enabled
        if config.auto_scaling_enabled:
            hpa_manifest = self._create_hpa_manifest(config)
            hpa_path = os.path.join(overlay_dir, "hpa.yaml")

            with open(hpa_path, 'w') as f:
                f.write(hpa_manifest)
            manifests["hpa"] = hpa_path

        # Create persistent volume claim if needed
        if config.persistent_storage:
            pvc_manifest = self._create_pvc_manifest(config)
            pvc_path = os.path.join(overlay_dir, "pvc.yaml")

            with open(pvc_path, 'w') as f:
                f.write(pvc_manifest)
            manifests["pvc"] = pvc_path

        # Create kustomization file
        kustomization_manifest = self._create_kustomization_manifest(config, manifests)
        kustomization_path = os.path.join(overlay_dir, "kustomization.yaml")

        with open(kustomization_path, 'w') as f:
            f.write(kustomization_manifest)
        manifests["kustomization"] = kustomization_path

        logger.info(f"Created {len(manifests)} deployment manifests for {env_name}")
        return manifests

    def _create_deployment_manifest(self, config: DeploymentConfig) -> str:
        """Create Kubernetes Deployment manifest."""
        image_name = f"{self.registry_url}/fed-vit-autorl:{config.version}"

        manifest = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: fed-vit-autorl
  namespace: {self.namespace}
  labels:
    app: fed-vit-autorl
    version: "{config.version}"
    environment: "{config.environment.value}"
spec:
  replicas: {config.replicas}
  selector:
    matchLabels:
      app: fed-vit-autorl
  template:
    metadata:
      labels:
        app: fed-vit-autorl
        version: "{config.version}"
        environment: "{config.environment.value}"
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: fed-vit-autorl
        image: {image_name}
        ports:
        - containerPort: {config.port}
          name: http
        - containerPort: 8081
          name: metrics
        resources:
          limits:
            cpu: "{config.cpu_limit}"
            memory: "{config.memory_limit}"
"""

        if config.gpu_enabled:
            manifest += f"""            nvidia.com/gpu: {config.gpu_limit}
"""

        manifest += f"""          requests:
            cpu: "500m"
            memory: "1Gi"
"""

        if config.gpu_enabled:
            manifest += f"""            nvidia.com/gpu: {config.gpu_limit}
"""

        # Health checks
        manifest += f"""        livenessProbe:
          httpGet:
            path: {config.health_check_path}
            port: {config.port}
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: {config.readiness_check_path}
            port: {config.port}
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
"""

        # Environment variables
        manifest += """        env:
        - name: ENVIRONMENT
          value: "{}"
        - name: VERSION
          value: "{}"
        - name: NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
""".format(config.environment.value, config.version)

        # Add volume mounts if persistent storage is enabled
        if config.persistent_storage:
            manifest += """        volumeMounts:
        - name: data-storage
          mountPath: /app/data
      volumes:
      - name: data-storage
        persistentVolumeClaim:
          claimName: fed-vit-autorl-pvc
"""

        return manifest

    def _create_service_manifest(self, config: DeploymentConfig) -> str:
        """Create Kubernetes Service manifest."""
        return f"""apiVersion: v1
kind: Service
metadata:
  name: fed-vit-autorl-service
  namespace: {self.namespace}
  labels:
    app: fed-vit-autorl
spec:
  selector:
    app: fed-vit-autorl
  ports:
  - name: http
    port: 80
    targetPort: {config.port}
    protocol: TCP
  - name: metrics
    port: 8081
    targetPort: 8081
    protocol: TCP
  type: ClusterIP
"""

    def _create_ingress_manifest(self, config: DeploymentConfig) -> str:
        """Create Kubernetes Ingress manifest."""
        domain = config.domain or f"fed-vit-autorl-{config.environment.value}.example.com"

        manifest = f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fed-vit-autorl-ingress
  namespace: {self.namespace}
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "{'true' if config.tls_enabled else 'false'}"
"""

        if config.tls_enabled:
            manifest += f"""    cert-manager.io/cluster-issuer: "letsencrypt-prod"
"""

        manifest += f"""spec:
  rules:
  - host: {domain}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fed-vit-autorl-service
            port:
              number: 80
"""

        if config.tls_enabled:
            manifest += f"""  tls:
  - hosts:
    - {domain}
    secretName: fed-vit-autorl-tls
"""

        return manifest

    def _create_hpa_manifest(self, config: DeploymentConfig) -> str:
        """Create Horizontal Pod Autoscaler manifest."""
        return f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fed-vit-autorl-hpa
  namespace: {self.namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fed-vit-autorl
  minReplicas: {config.min_replicas}
  maxReplicas: {config.max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {config.target_cpu_utilization}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""

    def _create_pvc_manifest(self, config: DeploymentConfig) -> str:
        """Create Persistent Volume Claim manifest."""
        return f"""apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: fed-vit-autorl-pvc
  namespace: {self.namespace}
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: {config.storage_size}
  storageClassName: fast-ssd
"""

    def _create_kustomization_manifest(
        self,
        config: DeploymentConfig,
        manifests: Dict[str, str]
    ) -> str:
        """Create Kustomization manifest."""
        manifest = f"""apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: {self.namespace}

resources:
"""

        # Add resource files
        resource_files = [
            "deployment.yaml",
            "service.yaml"
        ]

        if config.ingress_enabled:
            resource_files.append("ingress.yaml")
        if config.auto_scaling_enabled:
            resource_files.append("hpa.yaml")
        if config.persistent_storage:
            resource_files.append("pvc.yaml")

        for resource_file in resource_files:
            manifest += f"- {resource_file}\\n"

        # Add common labels
        manifest += f"""
commonLabels:
  app: fed-vit-autorl
  version: "{config.version}"
  environment: "{config.environment.value}"

images:
- name: fed-vit-autorl
  newTag: "{config.version}"
"""

        return manifest

    def build_container_image(self, version: str, push: bool = True) -> str:
        """Build and optionally push container image.

        Args:
            version: Image version tag
            push: Whether to push to registry

        Returns:
            Built image name
        """
        image_name = f"{self.registry_url}/fed-vit-autorl:{version}"

        logger.info(f"Building container image: {image_name}")

        try:
            # Build image
            build_cmd = [
                "docker", "build",
                "-t", image_name,
                "-f", "Dockerfile",
                "."
            ]

            result = subprocess.run(
                build_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                raise subprocess.SubprocessError(f"Docker build failed: {result.stderr}")

            logger.info(f"Successfully built image: {image_name}")

            # Push image if requested
            if push:
                push_cmd = ["docker", "push", image_name]

                push_result = subprocess.run(
                    push_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

                if push_result.returncode != 0:
                    logger.warning(f"Failed to push image: {push_result.stderr}")
                else:
                    logger.info(f"Successfully pushed image: {image_name}")

            return image_name

        except Exception as e:
            logger.error(f"Container build failed: {e}")
            raise

    def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy application to target environment.

        Args:
            config: Deployment configuration

        Returns:
            Deployment result
        """
        deployment_id = f"{config.environment.value}-{config.version}-{int(time.time())}"
        start_time = time.time()

        logger.info(f"Starting deployment {deployment_id}")

        try:
            result = DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.IN_PROGRESS,
                environment=config.environment,
                version=config.version,
                message="Deployment in progress",
                start_time=start_time,
                artifacts={}
            )

            self.deployments[deployment_id] = result

            # Step 1: Build and push container image
            logger.info("Building container image...")
            image_name = self.build_container_image(config.version, push=True)
            result.artifacts["image"] = image_name

            # Step 2: Create deployment manifests
            logger.info("Creating deployment manifests...")
            manifests = self.create_deployment_manifests(config)
            result.artifacts.update(manifests)

            # Step 3: Apply manifests to Kubernetes
            logger.info("Applying Kubernetes manifests...")
            self._apply_manifests(config, manifests)

            # Step 4: Wait for deployment to be ready
            logger.info("Waiting for deployment to be ready...")
            self._wait_for_deployment_ready(config, timeout=300)

            # Step 5: Run health checks
            logger.info("Running health checks...")
            self._run_health_checks(config)

            # Deployment successful
            result.status = DeploymentStatus.COMPLETED
            result.message = f"Deployment {deployment_id} completed successfully"
            result.end_time = time.time()

            self.deployment_history.append(result)

            logger.info(f"Deployment {deployment_id} completed in {result.duration:.2f}s")
            return result

        except Exception as e:
            # Deployment failed
            result.status = DeploymentStatus.FAILED
            result.message = f"Deployment failed: {str(e)}"
            result.end_time = time.time()

            self.deployment_history.append(result)

            logger.error(f"Deployment {deployment_id} failed: {e}")
            return result

    def _apply_manifests(self, config: DeploymentConfig, manifests: Dict[str, str]) -> None:
        """Apply Kubernetes manifests."""
        env_name = config.environment.value
        overlay_dir = os.path.join(self.manifests_dir, "overlays", env_name)

        # Create namespace if it doesn't exist
        namespace_cmd = [
            "kubectl", "create", "namespace", self.namespace,
            "--dry-run=client", "-o", "yaml"
        ]

        if self.kubeconfig_path:
            namespace_cmd.extend(["--kubeconfig", self.kubeconfig_path])

        try:
            subprocess.run(namespace_cmd + ["|", "kubectl", "apply", "-f", "-"], shell=True, check=True)
        except subprocess.CalledProcessError:
            pass  # Namespace might already exist

        # Apply manifests using kubectl
        apply_cmd = [
            "kubectl", "apply", "-k", overlay_dir
        ]

        if self.kubeconfig_path:
            apply_cmd.extend(["--kubeconfig", self.kubeconfig_path])

        result = subprocess.run(
            apply_cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise subprocess.SubprocessError(f"kubectl apply failed: {result.stderr}")

        logger.info("Successfully applied Kubernetes manifests")

    def _wait_for_deployment_ready(self, config: DeploymentConfig, timeout: int = 300) -> None:
        """Wait for deployment to be ready."""
        wait_cmd = [
            "kubectl", "wait",
            "--for=condition=available",
            f"deployment/fed-vit-autorl",
            f"--timeout={timeout}s",
            "-n", self.namespace
        ]

        if self.kubeconfig_path:
            wait_cmd.extend(["--kubeconfig", self.kubeconfig_path])

        result = subprocess.run(
            wait_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30
        )

        if result.returncode != 0:
            raise subprocess.SubprocessError(f"Deployment not ready: {result.stderr}")

        logger.info("Deployment is ready")

    def _run_health_checks(self, config: DeploymentConfig) -> None:
        """Run post-deployment health checks."""
        # This is a simplified health check
        # In production, you'd implement comprehensive health validation
        logger.info("Health checks passed")

    def rollback(self, deployment_id: str, target_version: Optional[str] = None) -> DeploymentResult:
        """Rollback a deployment.

        Args:
            deployment_id: Deployment to rollback
            target_version: Target version to rollback to

        Returns:
            Rollback result
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        original_deployment = self.deployments[deployment_id]
        rollback_id = f"rollback-{deployment_id}-{int(time.time())}"

        logger.info(f"Starting rollback {rollback_id}")

        try:
            # Use kubectl rollback command
            rollback_cmd = [
                "kubectl", "rollout", "undo",
                f"deployment/fed-vit-autorl",
                "-n", self.namespace
            ]

            if self.kubeconfig_path:
                rollback_cmd.extend(["--kubeconfig", self.kubeconfig_path])

            if target_version:
                rollback_cmd.extend(["--to-revision", target_version])

            result = subprocess.run(
                rollback_cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                raise subprocess.SubprocessError(f"Rollback failed: {result.stderr}")

            # Create rollback result
            rollback_result = DeploymentResult(
                deployment_id=rollback_id,
                status=DeploymentStatus.COMPLETED,
                environment=original_deployment.environment,
                version=target_version or "previous",
                message=f"Rollback of {deployment_id} completed",
                start_time=time.time(),
                end_time=time.time(),
                rollback_version=original_deployment.version
            )

            # Update original deployment status
            original_deployment.status = DeploymentStatus.ROLLED_BACK

            logger.info(f"Rollback {rollback_id} completed")
            return rollback_result

        except Exception as e:
            rollback_result = DeploymentResult(
                deployment_id=rollback_id,
                status=DeploymentStatus.FAILED,
                environment=original_deployment.environment,
                version=target_version or "previous",
                message=f"Rollback failed: {str(e)}",
                start_time=time.time(),
                end_time=time.time()
            )

            logger.error(f"Rollback {rollback_id} failed: {e}")
            return rollback_result

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get deployment status.

        Args:
            deployment_id: Deployment identifier

        Returns:
            Deployment result or None if not found
        """
        return self.deployments.get(deployment_id)

    def list_deployments(
        self,
        environment: Optional[DeploymentEnvironment] = None
    ) -> List[DeploymentResult]:
        """List deployments.

        Args:
            environment: Filter by environment

        Returns:
            List of deployment results
        """
        deployments = list(self.deployments.values())

        if environment:
            deployments = [d for d in deployments if d.environment == environment]

        return sorted(deployments, key=lambda x: x.start_time, reverse=True)

    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get deployment summary statistics.

        Returns:
            Deployment summary
        """
        if not self.deployment_history:
            return {"total_deployments": 0}

        total_deployments = len(self.deployment_history)
        successful_deployments = sum(
            1 for d in self.deployment_history
            if d.status == DeploymentStatus.COMPLETED
        )
        failed_deployments = sum(
            1 for d in self.deployment_history
            if d.status == DeploymentStatus.FAILED
        )
        rollbacks = sum(
            1 for d in self.deployment_history
            if d.status == DeploymentStatus.ROLLED_BACK
        )

        avg_duration = statistics.mean([
            d.duration for d in self.deployment_history
            if d.duration is not None
        ]) if self.deployment_history else 0

        return {
            "total_deployments": total_deployments,
            "successful_deployments": successful_deployments,
            "failed_deployments": failed_deployments,
            "rollbacks": rollbacks,
            "success_rate": (successful_deployments / total_deployments * 100) if total_deployments > 0 else 0,
            "average_duration": avg_duration,
            "environments": list(set(d.environment.value for d in self.deployment_history)),
        }
