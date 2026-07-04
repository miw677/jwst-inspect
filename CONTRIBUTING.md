# Contributing

One main project, one fork per team. The main project is
`gitlab.com/nvidia-harvard/jwst_inspect`; your team works in its fork
(`jwst_inspect-digital_twin`, `jwst_inspect-benchmark`, `jwst_inspect-autonomous`).
Full guide with commands: [docs/gitlab_workflow.md](docs/gitlab_workflow.md).

Who does what:

- Members push and pull on their own team's fork (per-person deploy key, no
  GitLab account needed): branch from the fork's `main`, commit, push
  `<username>/<topic>`, ask the representative for review. Members edit only
  their own team's tree (`<team>/src`, `<team>/sbatch`, `<team>/interface`).
- The representative (one GitLab seat per team) reviews and merges member
  branches into the fork's `main`, opens the merge request from the fork to
  the main project (weekly, and on every interface change), and syncs the fork
  from the main project after every upstream merge.
- The admin reviews and merges main-project merge requests and owns the shared
  root: `README.md`, `data/`, `docs/`, `assets/`, `bin/`, seats, deploy keys,
  and the size budget.

Hard rules:

- The main project's `main` moves only by merge request; a fork's `main` moves
  only by its representative or the admin; members push topic branches only.
- No secrets, caches, run outputs, containers, checkpoints, or generated
  datasets in git; `.gitignore` covers the known offenders. Big artifacts have
  homes on the workstation (`/data/shared/{assets,datasets,checkpoints}`).
- Binaries go through Git LFS (`.gitattributes`); anything over 100 MB or any
  new dataset goes through the admin first (the project has a hard 10 GiB
  free-tier cap).
