/usr/src/tensorrt/bin/trtexec --onnx=yolo11_robust.onnx \
          --saveEngine=yolo11_robust.engine \
          --fp16 \
          --workspace=1024 \
          --buildOnly \
          --verbose