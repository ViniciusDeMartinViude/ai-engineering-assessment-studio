# AI Engineering Assessment Studio

AI Engineering date-fruit training and assessment workstation. M1 currently provides a PySide6 desktop shell with navigation across the planned application pages, a candidate workspace, append-only local event logs, and idempotent server-receipt acknowledgement storage.

The M1 shell does not connect to an organizer server yet. Local events remain pending until a future organizer integration records a server acknowledgement. No calibration, dataset, vision, robot, training, IDE, AI gateway, or submission workflow is implemented in this milestone.

Read [Architecture v1](docs/ARCHITECTURE_V1.md) and [AGENTS.md](AGENTS.md) for the project boundaries and acceptance gates.

## Windows setup

Use Anaconda Prompt or another shell where `conda` is initialized. The project baseline targets Python 3.12:

```bat
cd /d C:\Projects\ai-engineering-assessment-studio
conda create -n ai-assessment-studio python=3.12 -y
conda activate ai-assessment-studio
python -m pip install -e ".[gui]"
```

If the repository is in a different folder, replace the `cd` path. The editable install makes the `ai_assessment` package available to the commands below.

## Run M1

Create or reopen the default `C014` workspace and launch the desktop shell:

```bat
python -m ai_assessment --workspace candidate_workspaces\C014
```

The candidate ID is only needed when creating a new workspace. When reopening an existing workspace, omit `--candidate-id` or provide the ID already stored in `session.json`. A different explicit ID is rejected.

Run the non-GUI workspace check:

```bat
python -m ai_assessment --headless-check --workspace candidate_workspaces\C014
```

Run the M1 tests:

```bat
python -m unittest discover -s tests -v
```

Events are stored in `candidate_workspaces\C014\logs\events.jsonl`; server receipts are stored in `candidate_workspaces\C014\logs\event_acks.jsonl`.

The sample package has no live robot credentials, API keys, copyrighted dataset payload, or candidate data.
