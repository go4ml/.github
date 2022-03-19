
Set of packages designed to simplify to do machine learning with Golang. 

Like Neural Networks
```golang
import (
	"go4ml.xyz/nn"
	"go4ml.xyz/nn/mx"
)

var mnistConv0 = nn.Sequence(
	nn.Convolution{Channels: 24, Kernel: mx.Dim(3, 3), Activation: nn.ReLU},
	nn.MaxPool{Kernel: mx.Dim(2, 2), Stride: mx.Dim(2, 2)},
	nn.Convolution{Channels: 32, Kernel: mx.Dim(5, 5), Activation: nn.ReLU, BatchNorm: true},
	nn.MaxPool{Kernel: mx.Dim(2, 2), Stride: mx.Dim(2, 2)},
	nn.FullyConnected{Size: 32, Activation: nn.Swish, BatchNorm: true, Dropout: 0.33},
	nn.FullyConnected{Size: 10, Activation: nn.Softmax})
```

