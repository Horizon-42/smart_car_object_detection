/usr/src/tensorrt/bin/trtexec --onnx=yolo11_robust2.onnx \
          --saveEngine=yolo11_robust2.engine \
          --fp16 \
          --workspace=1024 \
          --buildOnly \
          --verbose