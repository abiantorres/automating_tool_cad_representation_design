# Automating Tool CAD Representation Design

A Python library for CAD representation design automation.

## Installation

### Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/automating_tool_cad_representation_design.git
cd automating_tool_cad_representation_design

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### From PyPI (when published)

```bash
pip install cadlib
```

## Usage

```python
from cadlib import CADSequence, vec2CADsolid

# Example usage
cad_sequence = CADSequence.from_vector(your_vector)
solid = vec2CADsolid(your_vector)
```

## Dependencies

- numpy
- matplotlib  
- trimesh
- pythonocc-core