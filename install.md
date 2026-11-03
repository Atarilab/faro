## 1. Clone FARO
```bash
git clone https://github.com/Atarilab/faro.git
cd faro
```
```bash
git submodule update --init --recursive
```
## 2. Make Conda Env

For Linux x86-64, use the explicit lockfile captured from the tested environment:

```bash
conda create -n faro_env --file conda-linux-64.lock
conda activate faro_env
```

The lockfile fixes all Conda packages, including transitive dependencies, to
exact versions, builds, download URLs, and checksums. Conda installs these
artifacts without solving for newer versions. `environment.yml` records exact
direct dependency versions; creating an environment from that YAML still
resolves transitive dependencies and can produce a different environment.

This snapshot requires Linux x86-64, glibc 2.34 or newer, and a CPU compatible
with the locked Skylake / x86-64-v3 packages. Generate a separate lockfile for
other platforms. The lock covers Conda packages; pip-installed packages and
the local Acados, Moto, and FARO builds remain separate steps below.

```bash
conda env config vars set ACADOS_SOURCE_DIR="$PWD/extern/acados"
conda env config vars set LD_LIBRARY_PATH="$PWD/extern/acados/lib:$CONDA_PREFIX/lib"

conda deactivate
conda activate faro_env
```
## 3. Build Acados
```bash
cd extern/acados
mkdir -p build
cd build
cmake -DBLASFEO_TARGET=X64_AUTOMATIC -DHPIPM_TARGET=AVX -DACADOS_WITH_OPENMP=ON -DACADOS_NUM_THREADS=10 ..
make install -j4
```
```bash
pip install $ACADOS_SOURCE_DIR/interfaces/acados_template
```
```bash
cd ../../..
```
## 4. Build Moto

Use the pinned Moto submodule commit: FARO requires its `casadi_manifold` and
`codegen_context` APIs. After updating the submodule, rebuild and reinstall Moto
in the environment used to run FARO.

```bash
cd extern/moto

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF \
  -DWITH_NATIVE_OPT=ON

cmake --build build --clean-first -j4
cmake --install build
```
```bash
cd ../..
```
## 5. Install FARO
From the `faro` repository root:
```bash
pip install -e .
```
## 6. Test install
```bash
python - <<'PY'
import moto
from faro.tasks.ik_task import IkTask
from faro.tasks.to_task import ToTask
from scenes.common.registry import make_scene_with_state

required = ("stage", "casadi_manifold", "codegen_context")
missing = [name for name in required if not hasattr(moto, name)]
assert not missing, f"Rebuild and reinstall extern/moto in this environment: missing {missing}"
scene, state = make_scene_with_state("example_scene", "default")
print(f"Loaded {scene.name} with {len(state.robots)} robot(s).")
PY
```

Check all scene presets and bundled visualization assets without compiling
solvers or opening a viewer:

```bash
python -m unittest discover -s tests -v
```

## 7. Refresh the Conda lock after an intentional upgrade

After testing the upgraded environment, export its exact Conda packages
automatically using [Conda's built-in explicit export](https://docs.conda.io/projects/conda/en/stable/commands/list.html):

```bash
conda list -n faro_env --explicit --md5 > conda-linux-64.lock
```

Replace `faro_env` with the tested environment's name if different. Update the
direct versions in `environment.yml` to match and commit both files together.
The MD5 option is supported by the Conda version used to capture this snapshot.
Keep using the committed lockfile for normal installations; regenerate it only
when deliberately changing dependencies. Package artifacts must remain
available from their recorded URLs or a local package cache.
