# Validation Evidence Checklist

Do not paste invented output into the final PR. Run each command and attach the real terminal output or screenshots.

## Local Docker

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/checkpoints:/app/checkpoints" mlops-train:v1
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 -v "$(pwd)/checkpoints:/app/checkpoints:ro" mlops-serve:v1
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

Capture the two successful builds, JSON training metrics, checkpoint-saved event, healthy API response, and prediction response.

## Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/training-job.yaml
kubectl logs -f job/model-training -n ml-training
kubectl wait --for=condition=complete job/model-training -n ml-training --timeout=30m
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl get pods,jobs,deployments,services,hpa -n ml-training
kubectl describe deployment model-serving -n ml-training
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

Capture manifest application, completed Job, training logs, two ready serving pods, Deployment probe configuration, HPA, health response, and prediction response.

