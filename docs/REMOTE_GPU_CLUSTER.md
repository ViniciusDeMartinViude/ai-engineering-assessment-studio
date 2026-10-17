# Remote GPU training integration (pilot)

This branch adds a **Remote GPU Cluster** panel to Training, retaining local M6 training unchanged.

## Architecture
Candidate desktop -> organizer gateway -> GPU management API -> worker nodes.

**Never put the management API_KEY or WORKER_KEY on student PCs.** The gateway validates a separate per-candidate bearer token and stamps student_id from its own token registry. Configure HTTPS, firewall controls and individual tokens before connecting untrusted candidate workstations.

## Setup
Deploy the gateway app from `integrations/gpu_gateway.py` on the management PC (or as another organizer-managed service that can reach the management API at localhost). It needs `fastapi uvicorn requests` plus:
- `CLUSTER_MANAGER_URL=http://127.0.0.1:8000`
- `CLUSTER_API_KEY` from the management host's private .env
- `GATEWAY_TOKENS_FILE` an organizer-only JSON file mapping token -> candidate ID (C###).
- `GATEWAY_DATASETS=dates-v16` comma-separated, pre-staged dataset IDs.

Example: `uvicorn integrations.gpu_gateway:app --host 127.0.0.1 --port 8443`.
Use an authenticated TLS reverse proxy to make the gateway available over the LAN; do not bind an unauthenticated gateway publicly.

In Assessment Studio's Training page, enter the gateway URL, student-issued token, dataset ID, model and training parameters. Submit, refresh, cancel, download ZIP. Network calls are asynchronous.

**Pilot limitation:** This uses a dataset already staged by organizers on the GPU manager. It does not upload each student's M3 four-class subset, does not convert remote artifacts to native M6 Results records, and does not expose candidate-independent cluster controls. Before competition release, implement trusted candidate authentication, subset upload with hash verification, results import and progress polling.

## Quick tests
1. Start with an organizer test candidate token and pre-staged dataset.
2. Submit a short job through the Studio, confirm that manager job has the correct student ID.
3. Refresh until completed, download the artifact ZIP into the candidate workspace.
4. Confirm that a different candidate token cannot read or cancel that job.
5. Confirm local Training still works.
