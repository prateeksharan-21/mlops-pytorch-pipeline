from pathlib import Path

import yaml

K8S_DIR = Path(__file__).parents[1] / "k8s"


def load_manifest(filename: str) -> dict:
    with (K8S_DIR / filename).open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    assert isinstance(manifest, dict)
    return manifest


def test_training_job_has_required_config_storage_and_resources() -> None:
    job = load_manifest("training-job.yaml")
    pod_spec = job["spec"]["template"]["spec"]
    trainer = pod_spec["containers"][0]

    assert trainer["resources"]["requests"] == {"cpu": "2", "memory": "4Gi"}
    assert trainer["resources"]["limits"] == {"cpu": "2", "memory": "4Gi"}
    assert {mount["mountPath"] for mount in trainer["volumeMounts"]} == {
        "/app/configs",
        "/app/data",
        "/app/checkpoints",
    }
    assert pod_spec["restartPolicy"] == "Never"


def test_serving_deployment_has_rollout_probes_and_read_only_model() -> None:
    deployment = load_manifest("serving-deployment.yaml")
    spec = deployment["spec"]
    container = spec["template"]["spec"]["containers"][0]

    assert spec["replicas"] == 2
    assert spec["strategy"]["rollingUpdate"] == {
        "maxSurge": 1,
        "maxUnavailable": 0,
    }
    assert container["livenessProbe"]["periodSeconds"] == 10
    assert container["livenessProbe"]["failureThreshold"] == 3
    assert container["readinessProbe"]["initialDelaySeconds"] == 15
    assert container["readinessProbe"]["periodSeconds"] == 5
    assert container["resources"]["requests"] == {
        "cpu": "500m",
        "memory": "1Gi",
    }
    assert container["resources"]["limits"] == {"cpu": "1", "memory": "2Gi"}
    checkpoint_mount = next(
        mount
        for mount in container["volumeMounts"]
        if mount["mountPath"] == "/app/checkpoints"
    )
    assert checkpoint_mount["readOnly"] is True


def test_service_hpa_and_embedded_training_config() -> None:
    service = load_manifest("serving-service.yaml")
    hpa = load_manifest("hpa.yaml")
    configmap = load_manifest("configmap.yaml")
    training_config = yaml.safe_load(configmap["data"]["training_config.yaml"])

    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["ports"][0] == {
        "name": "http",
        "port": 80,
        "targetPort": 8080,
    }
    assert hpa["spec"]["minReplicas"] == 2
    assert hpa["spec"]["maxReplicas"] == 5
    assert hpa["spec"]["metrics"][0]["resource"]["target"][
        "averageUtilization"
    ] == 70
    assert training_config["data"]["dataset"] == "cifar10"
    assert training_config["output"]["checkpoint_dir"] == "/app/checkpoints"
