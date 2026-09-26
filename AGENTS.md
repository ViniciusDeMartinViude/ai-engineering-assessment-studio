# Development instructions — AI Engineering Assessment Studio

Read `docs/ARCHITECTURE_V1.md` before implementing a module. Preserve the source prototypes unchanged when importing to `legacy/` and record their provenance in `docs/SOURCE_INVENTORY.md`.

- Target Windows 11, Python 3.12 and PySide6 for the desktop baseline; pin package versions after testing on the official image.
- UI modules depend on services/contracts, not on one another's widgets. Keep capture, YOLO, training, HTTP and child processes off the UI thread.
- Display a light, legible interface with sensible numeric precision; store matrices and measurements at full precision.
- Resolve all candidate file writes within the candidate workspace. Keep source dataset read-only and hidden assessment material server-side.
- Simulator and real MaxArm adapters expose only verified common capabilities. Do not assume XYZ moves, home, or stop exist in firmware based on a proposal.
- Never store the provider API key on a candidate PC. Route AI through organizer gateway and record usage centrally.
- Treat the application as UX and evidence capture, not a network security boundary. No secret, candidate data, model weights, or dataset dump in Git.
- Test contracts and material risks, especially coordinate transforms, class ID remapping, robot errors, event idempotence, and budget concurrency.
- For each change, state what changed, how it was verified, and remaining hardware/environment dependencies.

Do not change official scoring thresholds without an approved pilot decision and a versioned rules update.
