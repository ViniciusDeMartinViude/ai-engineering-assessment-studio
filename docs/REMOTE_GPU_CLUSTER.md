# Direct remote GPU training (trusted-operator pilot)

Assessment Studio Training includes **Remote GPU Cluster** controls and retains local M6 training.

## Connect
Run the Studio from branch `feature/remote-gpu-cluster-training`. Enter the manager's LAN URL (for example `http://192.168.1.100:8000`) and the existing management API key. Select the pre-staged dataset ID `dates-v16`, model, epochs, image size and batch. Submit, Refresh, and Download selected result.

The desktop calls the v0.8 management API directly:
- POST /jobs (student_id from active workspace, mode=train, dataset/model/epochs/imgsz/batch)
- GET /jobs
- GET /jobs/{id}
- POST /jobs/{id}/cancel
- GET /jobs/{id}/artifact

Downloaded ZIP files land under `candidate_workspaces/C###/exports/remote_training/`.

## Security
**For organizer-operated trusted workstations only.** The existing manager key is administrator-grade and gives extensive access to the entire cluster. Displaying only the current candidate's jobs is UI filtering, **not authorization**. Do not distribute the key to competitors, bake it into an installer, store it in the workspace, or expose the manager to an untrusted LAN. Prefer a private management network and HTTPS/VPN. Before student deployment, implement scoped keys directly in manager API (or a centralized trust boundary).

## Known limits
The direct client references a dataset already staged on the GPU manager; it does not transmit selected M3 subsets. Downloaded artifacts are not automatically registered with Results/Vision. Validate exact API paths and responses against your deployed v0.8 manager; no physical cluster E2E verification has yet occurred.

## Dataset upload extension (in progress)
Training now lists valid M3 `dataset/subset-*` exports and offers **Upload selected subset**. The Studio packages the complete export (including `manifest.json`, `data.yaml`, train/val/test) as ZIP, computes SHA-256 and posts multipart `POST /datasets/upload` with `student_id`, `sha256`, `file`. The manager responds with `dataset_id`, which the UI fills into the dataset field for a subsequent training submission.

The implementation example is `integrations/cluster_dataset_upload_router.py`. **This is not automatically active**. The deployed cluster's `manager/app.py` must mount the route, register the uploaded dataset ID, and make its content available through the existing worker dataset download contract. No cluster source is present in this repository, so this wiring remains outstanding. Do not submit uploaded-dataset jobs until that integration is complete. Install `python-multipart` where this route is mounted.

Only trusted organizer workstations may use the direct cluster admin credential. The implementation still needs a real dataset/worker roundtrip test; archive extraction and M3 manifest content should be verified in the manager's deployment context before production use.
