#!/bin/bash

pthtop="$(cd "$(dirname "${0}")/../../../.." && pwd)"
source "${pthtop}"/manage/lib/params.sh
source "${pthtop}"/manage/lib/shared.sh
source "${pthcrr}"/params.sh

pthapp="${pthsrc}"/applcp
pthmdl="${pthapp}"/models

function getmdl {
  local mdltgt=${1} ; shift
  local mdlurl=${1} ; shift

  cd "${pthmdl}"
  if test ! -e "${mdltgt}"
  then
    if cnfrtn "import: ${mdltgt}: ${mdlurl}"
    then
      curl -LO "${mdlurl}" -o "${mdltgt}"
    fi
  fi
}

addimg ${imgtgt} "${cnfimg}" "${pthdoc}"

test -d "${pthapp}" || mkdir "${pthapp}"
test -d "${pthmdl}" || mkdir "${pthmdl}"

# https://huggingface.co/ggml-org/gemma-3n-E2B-it-GGUF
getmdl gemma-3n-E2B-it-Q8_0.gguf https://huggingface.co/ggml-org/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-Q8_0.gguf
