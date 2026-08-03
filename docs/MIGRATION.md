# Instance → robert-nfs migration

**NFS mount:** `/lambda/nfs/robert-nfs` (symlink `~/robert-nfs`)

## What's backed up here

```
cache-barrier-project/
  repos/           git clones (cache-barrier, kernel-compass, FlagGems)
  env/             pip freeze, versions, restore script
  artifacts/       paper PDF, arxiv tarball, key result JSON/figures
  docs/            SHUTDOWN_CHECKLIST, presentation prompts, meeting agenda
  backup.log       last rsync timestamp
```

## Restore on a new H100 instance

```bash
# 1. Mount NFS (Lambda: usually auto at /lambda/nfs/robert-nfs)
ln -sf /lambda/nfs/robert-nfs ~/robert-nfs

# 2. Clone or copy repos
cp -a ~/robert-nfs/cache-barrier-project/repos/* ~/
# OR: git clone from GitHub (repos are pushed; NFS is snapshot)

# 3. Python env
bash ~/robert-nfs/cache-barrier-project/env/restore_env.sh

# 4. kernel-compass submodule
cd ~/kernel-compass && git submodule update --init --recursive

# 5. Smoke tests
cd ~/kernel-compass && python3 tests/test_carm.py
cd ~/kernel-compass && python3 tests/test_validation.py   # needs GPU
cd ~/cache-barrier/paper && pdflatex -interaction=nonstopmode main.tex
```

## GitHub auth (2026-08-02)

Dedicated GitHub key lives on this NFS: `/lambda/nfs/robert-nfs/github-robert-ed25519`
(pub: `...IHokwH2oj+Q6StzvOYn5jldP7SttdKhMpR6bwRdLcwe1 robert-nfs-github`, added to
github.com account). All three repos have local `core.sshCommand` pointing at it, so
git push/pull works on any machine that mounts this NFS at `/lambda/nfs/robert-nfs`
— no per-machine key setup. (`lambda-robert-ed25519` at the NFS root is the Lambda
instance key, NOT a GitHub key.)

## Git remotes (source of truth)

| Repo | Remote |
|------|--------|
| cache-barrier | `git@github.com:zhan4808/cache-barrier.git` |
| kernel-compass | `git@github.com:zhan4808/kernel-compass.git` |
| FlagGems fork | `git@github.com:zhan4808/FlagGems.git` branch `fix-w8a16-inkernel-dequant` |

## Key measured constants (H100, 2026-06)

- C_eff ≈ 36 MB, HBM 3.146 TB/s, L2 5.331 TB/s
- t₀ graph 2.802 µs / eager 18.048 µs
- W8A8 wins 1.4–1.5× above cliff; 0.70× at 16 MB MLA
- MoE fixed W8A16: 2.7–3.0× at T≥512, no bf16 crossover to T=2048
