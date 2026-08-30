# Kubernetes Validation Evidence

Validated locally on 2026-08-30 (IST) with kind v0.32.0 and the officially supported, digest-pinned Kubernetes v1.34.8 node image:

```text
kindest/node:v1.34.8@sha256:02722c2dedddcfc00febf5d27fbeb9b7b2c14294c82109ff4a85d89ac9ba3256
```

Both locally built images were loaded directly into the `mlops-assignment` cluster:

```text
docker.io/library/mlops-serve  v1  e322f50ed9bf0  502MB
docker.io/library/mlops-train  v1  29954d8af5079  501MB
```

## Manifest validation

Every production and smoke manifest passed Kubernetes API server validation:

```text
configmap/training-config created (server dry run)
persistentvolumeclaim/ml-pipeline-pvc created (server dry run)
job.batch/model-training created (server dry run)
deployment.apps/model-serving created (server dry run)
service/model-serving created (server dry run)
horizontalpodautoscaler.autoscaling/model-serving created (server dry run)
configmap/training-config-smoke created (server dry run)
job.batch/model-training-smoke created (server dry run)
```

The node exposes 4 CPUs but only 3,908,732 KiB (about 3.73 GiB) allocatable memory. The rubric-compliant production Job correctly retains its required 2 CPU / 4 GiB requests, but that request cannot schedule on this node. Therefore, actual cluster execution used the clearly labeled synthetic smoke ConfigMap and Job with 500m CPU / 512 MiB requests. This is infrastructure evidence, not a CIFAR-10 model-quality claim.

## Training Job and persistent storage

```text
persistentvolumeclaim/ml-pipeline-pvc   Bound   10Gi   RWO   standard
job.batch/model-training-smoke          Complete   1/1
```

The Job produced structured logs and saved its checkpoint to the mounted PVC:

```json
{"architecture": "simple_cnn", "config_path": "/app/configs/training_config.yaml", "device": "cpu", "event": "training_started", "seed": 42}
{"epoch": 1, "event": "epoch_completed", "train_accuracy": 0.1074, "train_loss": 2.3378, "validation_accuracy": 0.1289, "validation_loss": 2.2974}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"best_validation_loss": 2.2974, "checkpoint_path": "/app/checkpoints/classifier_v1.pt", "event": "training_complete"}
```

## Serving rollout

The initial local rollout revealed that importing PyTorch concurrently could exceed the liveness probe's 30-second window. A startup probe was added to protect cold starts while preserving the assignment's required liveness and readiness settings. The corrected rolling update completed successfully:

```text
deployment.apps/model-serving   READY 2/2   UP-TO-DATE 2   AVAILABLE 2
service/model-serving           ClusterIP   port 80 -> targetPort 8080
```

```text
Replicas:               2 desired | 2 updated | 2 total | 2 available | 0 unavailable
RollingUpdateStrategy:  0 max unavailable, 1 max surge
Liveness:               GET /health, period=10s, failure=3
Readiness:              GET /health, delay=15s, period=5s
Startup:                GET /health, period=5s, failure=24
Checkpoint mount:       /app/checkpoints, read-only
```

## Service endpoint

Requests were sent through `kubectl port-forward svc/model-serving 8080:80 -n ml-training`:

```text
$ curl http://localhost:8080/health
{"status":"healthy","model":"loaded"}

$ curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
{"predicted_class":"dog","predicted_index":5,"probabilities":{"airplane":0.089273,"automobile":0.097024,"bird":0.09573,"cat":0.101076,"deer":0.103994,"dog":0.110076,"frog":0.09578,"horse":0.09568,"ship":0.108014,"truck":0.103353}}
```

## Autoscaling metrics

Metrics Server v0.8.0 was installed in the local test cluster. Because kind uses self-signed kubelet certificates, the testing-only `--kubelet-insecure-tls` flag was applied. The HPA then received real CPU data:

```text
NAME            REFERENCE                  TARGETS       MINPODS   MAXPODS   REPLICAS
model-serving   Deployment/model-serving   cpu: 0%/70%   2         5         2

model-serving-6d45bd9895-52psw   5m   217Mi
model-serving-6d45bd9895-dk6p6   4m   225Mi
```

