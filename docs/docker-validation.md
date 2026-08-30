# Docker Validation Evidence

Validated locally with Docker Desktop 29.5.2 on 2026-08-30 (IST). This is genuine command output from the workspace. The checkpoint below was produced with the synthetic infrastructure-only smoke configuration, not the production CIFAR-10 configuration.

## Image builds

```text
$ docker build -f docker/Dockerfile.train -t mlops-train:v1 .
#16 naming to docker.io/library/mlops-train:v1
#16 DONE

$ docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
#15 naming to docker.io/library/mlops-serve:v1
#15 DONE
```

The images use the official CPU-only PyTorch wheels:

```text
$ docker run --rm --entrypoint python mlops-train:v1 -c "import torch, torchvision; print('torch='+torch.__version__, 'torchvision='+torchvision.__version__, 'cuda='+str(torch.cuda.is_available()))"
torch=2.5.1+cpu torchvision=0.20.1+cpu cuda=False
```

```text
Image=mlops-train:v1 ID=sha256:3ccb95eb577719cfab3509f98b1a5926c0c8031867060e0ff3f4092c9a9191ea Size=500608787 User=root(default)
Image=mlops-serve:v1 ID=sha256:827374fd0cdec2950dc0b8f702be3866d3be109ea166d955110600731b813011 Size=501862957 User=appuser
```

## Containerized training smoke test

`configs/smoke_config.yaml` uses 1,024 synthetic training samples, 256 validation samples, one epoch, and `simple_cnn` to exercise the pipeline quickly.

```json
{"architecture": "simple_cnn", "config_path": "/app/configs/training_config.yaml", "device": "cpu", "event": "training_started", "seed": 42}
{"epoch": 1, "event": "epoch_completed", "train_accuracy": 0.1074, "train_loss": 2.3378, "validation_accuracy": 0.1289, "validation_loss": 2.2974}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"best_validation_loss": 2.2974, "checkpoint_path": "/app/checkpoints/classifier_v1.pt", "event": "training_complete"}
```

The mounted checkpoint was created successfully with a size of 1,161,073 bytes.

## Serving health and prediction

Docker reported the serving container as healthy and confirmed its non-root identity:

```text
Status=running Health=healthy User=appuser
uid=999(appuser) gid=999(appgroup) groups=999(appgroup)
```

```text
$ curl http://localhost:8080/health
{"status":"healthy","model":"loaded"}

$ curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
{"predicted_class":"dog","predicted_index":5,"probabilities":{"airplane":0.089273,"automobile":0.097024,"bird":0.09573,"cat":0.101076,"deer":0.103994,"dog":0.110076,"frog":0.09578,"horse":0.09568,"ship":0.108014,"truck":0.103353}}
```

The predicted label is not a model-quality claim because this checkpoint was trained on synthetic data. These results validate image upload, preprocessing, checkpoint loading, inference, probability serialization, health checks, volume mounts, and non-root serving.

## Kubernetes handoff

Kubernetes manifests and cluster validation evidence are intentionally scoped to the separate `feature/kubernetes` pull request.
