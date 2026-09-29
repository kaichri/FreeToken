#!/usr/bin/env bash
# Pre-push check for the mtp-qwen38 branch.
#
# Run before every push. It reports and refuses; it never rewrites the working tree, never
# discards a staged change, and never pushes on its own. A refusal is a list of things to
# look at, not a cleanup.
#
#   scripts/mtp_prepush_check.sh              # checks the branch against upstream/main
#   scripts/mtp_prepush_check.sh --staged    # also checks what is currently staged
#
# The three Windows working-tree patches are a standing local-only state on this box: they are
# deliberately uncommitted, and a commit that picks them up is a mistake worth stopping for.

set -uo pipefail

BASE="${MTP_PUSH_BASE:-upstream/main}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
MAX_FILE_MB="${MTP_PUSH_MAX_FILE_MB:-5}"
WINDOWS_PATCHES=(
  "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp"
  "python/freetoken/scheduler/config.py"
  "python/freetoken/server/args.py"
)

fail=0
note() { printf '  %s\n' "$*"; }
block() { printf 'BLOCK  %s\n' "$*"; fail=1; }

echo "pre-push check: branch $BRANCH against $BASE"

# ---------------------------------------------------------------- what is being pushed
if ! git rev-parse --verify --quiet "$BASE" >/dev/null; then
  echo "BLOCK  $BASE does not exist locally; fetch it first (git fetch upstream main)"
  exit 1
fi

commits="$(git log --oneline "$BASE"..HEAD)"
count="$(printf '%s' "$commits" | grep -c . || true)"
# The scan covers BOTH what is already committed and what is sitting in the index. A weight
# file is usually git add-ed and committed in one go, so scanning only the commit range would
# leave the most common way of leaking one unchecked -- the index is where it is visible first.
staged="$(git diff --cached --name-only)"
pushed="$(git diff --name-only "$BASE"..HEAD)"
scope="$(printf '%s\n%s\n' "$pushed" "$staged" | grep -v '^$' | sort -u)"
scoped="$(printf '%s' "$scope" | grep -c . || true)"
echo "  $count commit(s) to push, $(printf '%s' "$pushed" | grep -c . || true) file(s) committed" \
     "and $(printf '%s' "$staged" | grep -c . || true) staged ($scoped to check)"
[ "$count" -gt 0 ] || [ "$scoped" -gt 0 ] || note "  note: nothing to push, HEAD is already at $BASE"

# ---------------------------------------------------------------- secrets and tokens
# Scans the content being pushed, not the whole tree, so an unrelated pre-existing string in
# the upstream history does not block every push. The patterns are the ones that matter here:
# a HuggingFace token and a GitHub token are the two this box can plausibly leak.
secret_re='(hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[baprs]-[A-Za-z0-9-]{10,})'
# The committed content, plus the staged content: a token typed into a file and git add-ed is
# not in any commit yet, and a commit is exactly when it would get published.
leaks="$(git grep -nIE "$secret_re" "$BASE"..HEAD -- 2>/dev/null | head -20 || true)"
if [ -n "$staged" ]; then
  staged_leaks="$(git diff --cached -U0 | grep -nIE "$secret_re" | head -20 || true)"
  if [ -n "$staged_leaks" ]; then
    leaks="$leaks
$staged_leaks"
  fi
fi
if [ -n "$(printf '%s' "$leaks" | tr -d ' \n')" ]; then
  block "possible secret or token in what would be pushed:"
  printf '%s\n' "$leaks" | sed 's/^/        /'
else
  note "secrets/tokens: none in the pushed or staged content"
fi

# ---------------------------------------------------------------- weights and dumps
weight_re='\.(safetensors|pt|pth|bin|ckpt|gguf|onnx|nc|npz|h5)$'
weights="$(printf '%s\n' "$scope" | grep -IE "$weight_re" || true)"
if [ -n "$weights" ]; then
  block "model-weight-looking files in what would be pushed:"
  printf '%s\n' "$weights" | sed 's/^/        /'
else
  note "model weights (.safetensors/.pt/.bin/...): none"
fi

# _mtp/dump is the Phase-0 probe's dump directory, and the .json/.pt siblings the benchmarks
# and probes write. They hold model residuals and measured dumps, so they never belong in git.
dumps="$(printf '%s\n' "$scope" | grep -E '(^|/)_mtp/|(^|/)(dump|dumps)/|\.dump$|/(score|accept|ab|verify_cost)\.json$' || true)"
if [ -n "$dumps" ]; then
  block "benchmark dumps or probe output in what would be pushed:"
  printf '%s\n' "$dumps" | sed 's/^/        /'
else
  note "benchmark dumps (_mtp/dump, *.dump, run json): none"
fi

# ---------------------------------------------------------------- size and binaries
# Sizes come from the committed blobs, so a large file that is present in the working tree but
# not in the commit does not trip this. A staged file is measured on disk, which is the same
# thing a commit would publish.
big=""
while IFS= read -r f; do
  [ -n "$f" ] || continue
  if git cat-file -e "HEAD:$f" 2>/dev/null; then
    size="$(git cat-file -s "$(git rev-parse "HEAD:$f")" 2>/dev/null || echo 0)"
  elif [ -f "$f" ]; then
    size="$(wc -c <"$f" | tr -d ' ')"
  else
    continue
  fi
  if [ "$size" -gt $((MAX_FILE_MB * 1024 * 1024)) ]; then
    big="$big$f ($(printf '%.1f' "$(echo "$size" | awk '{print $1/1048576}')") MB)"$'\n'
  fi
done < <(printf '%s\n' "$scope")
if [ -n "$big" ]; then
  block "files over ${MAX_FILE_MB} MB in what would be pushed:"
  printf '%s' "$big" | sed 's/^/        /'
else
  note "file sizes: nothing over ${MAX_FILE_MB} MB"
fi

binary="$(printf '%s\n' "$scope" | while read -r f; do
  [ -n "$f" ] || continue
  if git cat-file -e "HEAD:$f" 2>/dev/null; then
    git diff --numstat "$BASE"..HEAD -- "$f" | awk -v F="$f" '$1 == "-" && $2 == "-" { print F }'
  elif [ -f "$f" ] && ! grep -qI . "$f" 2>/dev/null; then
    # not text, and not committed yet: staged and binary
    echo "$f"
  fi
done)"
if [ -n "$binary" ]; then
  block "binary files in what would be pushed (review each one):"
  printf '%s\n' "$binary" | sed 's/^/        /'
else
  note "binary files: none"
fi

# ---------------------------------------------------------------- the windows patches
# These three are uncommitted on purpose on this machine. If a staged change claims one of
# them, the commit is carrying a local portability fix that was never meant to be published.
for p in "${WINDOWS_PATCHES[@]}"; do
  if printf '%s\n' "$staged" | grep -qx "$p"; then
    block "a windows working-tree patch is staged: $p"
  fi
done
if printf '%s\n' "$staged" | grep -qx "python/freetoken/server/args.py"; then
  note "note: server/args.py also carries MTP changes; only the ZMQ hunks are local"
fi
dirty_windows="$(git diff --name-only | grep -Fx -f <(printf '%s\n' "${WINDOWS_PATCHES[@]}") || true)"
if [ -n "$dirty_windows" ]; then
  note "windows patches still uncommitted (expected, leave them): $(printf '%s' "$dirty_windows" | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- verdict
if [ "$fail" -ne 0 ]; then
  echo
  echo "BLOCKED. Nothing was pushed and nothing was changed. Look at the items above."
  exit 1
fi
echo "OK: nothing blocked. Push with:"
echo "  git push origin $BRANCH"
