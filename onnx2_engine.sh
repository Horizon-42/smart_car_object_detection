/usr/src/tensorrt/bin/trtexec --onnx=yolo26n.onnx \
          --saveEngine=yolo26n.engine \
          --fp16 \
          --workspace=2048 \
          --verbose