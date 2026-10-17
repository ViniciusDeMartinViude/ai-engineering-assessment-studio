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
