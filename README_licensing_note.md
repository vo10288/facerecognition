## License

This repository's source code is released under the MIT License. See `LICENSE`.

## Third-party software notices

This project depends on third-party libraries with their own licenses. See
`THIRD_PARTY_NOTICES.txt` for a practical summary and review the original upstream
licenses before redistribution.

## Important note about InsightFace models

This repository does **not** include InsightFace pretrained model files.

If you want to run the application offline, place the required model files in:

```text
insightface_local/
└─ models/
   └─ buffalo_l/
      ├─ 1k3d68.onnx
      ├─ 2d106det.onnx
      ├─ det_10g.onnx
      ├─ genderage.onnx
      └─ w600k_r50.onnx
```

### Why the models are not included

The InsightFace **code** and the InsightFace **pretrained models** are not the same thing.
Model files may be subject to separate upstream terms from the library source code.

Because of that, this repository does not bundle:
- InsightFace model ZIP files
- extracted `buffalo_l` model files
- packaged executables that already contain those model files

### Recommended redistribution approach

You can safely publish:
- the project source code
- build scripts
- documentation
- executables that expect the user to place the model files locally

Before redistributing model files themselves, verify the applicable upstream model terms.
