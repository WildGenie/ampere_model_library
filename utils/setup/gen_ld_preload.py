# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import re
import subprocess
from pathlib import Path


# generate the list of preload files
script_dir = os.path.dirname(os.path.realpath(__file__))
ld_preload = []

try:
    ld_preload.extend(
        str(path)
        for path in Path("/root").rglob("libgomp*")
        if ".so" in path.name
    )

except OSError as e:
    print(e)

try:
    ld_preload.extend(
        str(path) for path in Path("/").rglob("libgomp*") if ".so" in path.name
    )

except OSError as e:
    print(e)
    ld_preload.extend(
        str(path)
        for path in Path("/lib").rglob("libgomp*")
        if ".so" in path.name
    )

try:
    ld_preload.extend(str(path) for path in Path("/").rglob("libGLdispatch.so.0"))
except OSError as e:
    print(e)
    ld_preload.extend(
        str(path) for path in Path("/lib").rglob("libGLdispatch.so.0")
    )

# test the preload for errors
os.environ["LD_PRELOAD"] = ":".join(ld_preload)
test_cmd = ["python3", "-c", "'print(\"AML\")'"]

process = subprocess.Popen(test_cmd, stderr=subprocess.PIPE)
_, errors = process.communicate()
errors = errors.decode("utf-8")

start_pattern = "object '"
start_indices = [match.start() + len(start_pattern) for match in re.finditer(start_pattern, errors)]

end_pattern = "' from LD_PRELOAD cannot be preloaded"
end_indices = [match.start() for match in re.finditer(end_pattern, errors)]

for start, end in zip(start_indices, end_indices):
    ld_preload.remove(errors[start:end])

with open(os.path.join(script_dir, ".ld_preload"), "w") as f:
    f.write(":".join(ld_preload))
