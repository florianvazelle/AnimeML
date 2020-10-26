# Unity C++ Template

## Usage

### Build

Use the following command to build the library.

```bash
cd cpp
cmake -Bbuild
cmake --build build # generate a shared lib (.so or .dll)
```

Move this library to the corresponding folder in VoronoiUnity :
- `cpp/Assets/Plugins/Linux/x86_64` for Linux
- `unity/Assets/Plugins/Win/x86_64` for Windows

⚠️ Unity need to restart when we change the library.

## Requirements

- [cmake](https://cmake.org/) (>= 3.8)