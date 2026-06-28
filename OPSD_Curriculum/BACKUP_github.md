# GitHub 백업 안내 (OPSD_Curriculum)

연구 코드 백업 설정 메모. 최초 push: 2026-06-28.

## Repo
- **URL**: https://github.com/daymin1010/OPSD_Curriculum (Private)
- **로컬 git root**: `/scratch/lami2026/personal/jimin_2782/src/`
  (src 전체가 repo이고, whitelist `.gitignore`로 `OPSD_Curriculum/` + `OPSD_Cur`만 추적)
- **브랜치**: `main`

## 평소 업데이트 (코드 수정 후)
```bash
cd /scratch/lami2026/personal/jimin_2782/src
git add -A
git commit -m "수정 내용 요약"
git push
```
remote · 인증 · gitignore는 이미 설정 완료. 추가 작업 없음.

## 인증 (공용 계정 안전)
- repo 전용 **deploy key** (write access). 계정 전체 SSH 키와 무관.
- 비밀키: `/scratch/lami2026/personal/jimin_2782/.git_ssh/id_ed25519_opsd`
  (공용 `~/.ssh` 안 건드림. repo-local `core.sshCommand`로만 적용)
- 인증 확인:
  ```bash
  ssh -i /scratch/lami2026/personal/jimin_2782/.git_ssh/id_ed25519_opsd \
      -o IdentitiesOnly=yes -T git@github.com
  # -> "Hi daymin1010/OPSD_Curriculum! ..." 나오면 정상
  ```

## 무엇이 올라가고 무엇이 빠지는가
**포함**: 본인 작성 코드 전부(reasoning_pivot, analysis_qwen3_8b, training, labeling),
vendored `training/opsd_src/`(본인 env 수정 포함), stage manifest `*.json`, 보고서 `*.md`.

**제외** (`.gitignore`):
- 거대 데이터: `*.pt` (67GB) · `*.npy` (8.8GB) · `*.npz` · `*.safetensors` · `*.bin`
  · `*.parquet` · `*.jsonl` · `*.csv` · `*.tsv` (체크포인트 · activation · 라벨 덤프, 재생성 가능)
- 캐시 · 로그: `__pycache__/` · `wandb/` · `runs/` · `slurm-*.out/.err`
- siyan-zhao/OPSD 무수정 clone: `OPSD_original/` · `reasoning_pivot/dataset_decision/OPSD_repo/`
  (각자 `.git` 그대로, 필요하면 `git clone https://github.com/siyan-zhao/OPSD`)

## 주의
- **새 코드 폴더를 `src/` 최상위에 추가**하면 whitelist 때문에 자동 무시됨
  → `.gitignore`에 `!/새폴더/` 한 줄 추가해야 추적됨.
- **`opsd_src`**는 vendoring됨(자체 `.git` 제거 → 일반 파일). upstream 다시 받으려면 위 clone 명령 사용.
- manifest까지 포함해 repo ~267MB (옛 빌드 tiered/r2fix 비중 큼). 줄이려면 옛 `stages_*` 폴더 정리.
