# neat-walking-sim

A script set for training, visualising and experimenting with the NEAT(Neuroevolution of Augmenting Topologies) algorithm application in a traning of network for a bipedal walker. Project offers demo app for presentation purposes.

## Build and run

```text
> [!WARNING]
> Installing additional dependencies on your system may be required!!!
> Read all the occuring error messages during installation instalation process.
```

### 1. Download the repository
### 2. In the repository directory run
```bash
python -m venv .venv
```

#### For Linux/macOS:
```bash
source .venv/bin/activate
```

#### For Windows (CMD):
```
.venv\Scripts\activate.bat
```

For Windows (Powershell):
```
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the demo app
```bash
python demo.py
```