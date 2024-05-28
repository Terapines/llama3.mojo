from tensor import Tensor, TensorShape, rand
from algorithm import vectorize, parallelize
from math import abs, round
from sys.info import num_performance_cores
from memory import memset

var workers = num_performance_cores()
alias nelts_q8 = (4 * simdwidthof[Int8]())
alias nelts_q32 = (4 * simdwidthof[Int32]())
alias nelts_f32 = (4 * simdwidthof[Float32]())
alias NUM_CONFIG_HEADER_BYTES = 256


@value
struct TensorSlice[type: DType]:
    """A Reference to a tensor.

    Parameters:
        type: The data type of the tensor.

    """

    var _data: DTypePointer[type]
    var _shape: TensorShape

    fn __init__(inout self, ptr: DTypePointer[type], shape: TensorShape):
        self._data = ptr
        self._shape = shape

    fn __init__(inout self, t: Tensor[type], layer: Int) raises:
        var num_layer_elements = t.shape().num_elements() / t.dim(0)

        self._data = t.data().offset(layer * num_layer_elements)

        if t.rank() == 2:
            self._shape = TensorShape(t.dim(1))
        elif t.rank() == 3:
            self._shape = TensorShape(t.dim(1), t.dim(2))
        else:
            raise Error("unimplemented rank")

    fn __init__(inout self, t: Tensor[type], layer: Int, row: Int) raises:
        var num_layer_elements = t.shape().num_elements() / t.dim(0)
        var num_row_elements = num_layer_elements / t.dim(1)

        self._data = t.data().offset(
            layer * num_layer_elements + row * num_row_elements
        )

        if t.rank() == 3:
            self._shape = TensorShape(t.dim(2))
        else:
            raise Error("unimplemented rank")

    fn __getitem__(self, idx: Int) -> SIMD[type, 1]:
        return self._data.load[width=1](idx)

    fn load[width: Int](self, idx: Int) -> SIMD[type, width]:
        return self._data.load[width=width](idx)

    fn store[width: Int](self, idx: Int, value: SIMD[type, width]):
        self._data.store[width=width](idx, value)

    fn data(self) -> DTypePointer[type]:
        return self._data

    fn shape(self) -> TensorShape:
        return self._shape

    fn num_elements(self) -> Int:
        return self._shape.num_elements()
    
    fn dim(self, idx: Int) -> Int:
        return self._shape[idx]

    fn rank(self) -> Int:
        return self._shape.rank()

@value
struct QuantizedTensor:
    """An 8-bit quantized tensor."""

    var _quantized: Tensor[DType.int8]
    var _scale: Tensor[DType.float32]
    var _group_size: Int

    fn __init__(
        inout self,
        shape: TensorShape,
        group_size: Int,
    ):
        self._quantized = Tensor[DType.int8](shape)
        var num_scale_factors = self._quantized.num_elements() // group_size
        self._scale = Tensor[DType.float32](TensorShape(num_scale_factors))
        self._group_size = group_size

    fn dequantize(self, dequantized: TensorSlice[DType.float32]):
        """Dequantize the tensor into `dequantized`.

        Args:
            dequantized: The tensor to store the dequantized values.
        """
        var num_elements = self._quantized.num_elements()
        var num_groups = num_elements // self._group_size

        # iterate over the groups
        @parameter
        fn dequantize_group(group: Int):
            var scale_factor = self._scale[group]

            # dequantize the group
            @parameter
            fn _dequantize[simd_width: Int](i: Int):
                var quantized_lane = self._quantized.load[width=simd_width](
                    group * self._group_size + i
                ).cast[DType.float32]()
                dequantized.store[width=simd_width](
                    group * self._group_size + i, quantized_lane * scale_factor
                )

            vectorize[_dequantize, nelts_q8](self._group_size)

        parallelize[dequantize_group](num_groups, num_groups)

    fn dequantize_naive(self, dequantized: TensorSlice[DType.float32]):
        """Dequantize the tensor into `dequantized`. Uses a naive implementation.

        Args:
            dequantized: The tensor to store the dequantized values.
        """
        var num_elements = self._quantized.num_elements()

        for i in range(num_elements):
            var group = i // self._group_size
            var scale_factor = self._scale[group]

            var quantized_lane = self._quantized.load[width=1](i).cast[DType.float32]()
            dequantized.store[width=1](i, quantized_lane * scale_factor)


    fn quantize(inout self, dequantized: TensorSlice[DType.float32]) raises:
        """Quantize the `dequantized` and update self.

        Args:
            dequantized: A tensor with float32 values to quantize.
        """

        if dequantized.shape() != self._quantized.shape():
            raise Error("shape mismatch, expected: " + str(self._quantized.shape()) + ", got: " + str(dequantized.shape()))

        var num_elements = dequantized.num_elements()

        if num_elements % self._group_size != 0:
            raise Error("number of elements must be a multiple of group size")

        var num_groups = num_elements // self._group_size

        var Q_MAX: Float32 = 127.0

        @parameter
        fn quantize_group(group: Int):
            var wmax: Float32 = 0.0

            @parameter
            fn _find_wmax[simd_width: Int](i: Int):
                var dequantized_lane = dequantized.load[width=simd_width](
                    group * self._group_size + i
                )
                var local_wmax = abs(dequantized_lane).reduce_max()
                if local_wmax > wmax:
                    wmax = local_wmax

            vectorize[_find_wmax, nelts_f32](self._group_size)

            var scale_factor = wmax / Q_MAX
            self._scale.store[width=1](group, scale_factor)

            # calculate and write back the quantized values
            @parameter
            fn _quantize[simd_width: Int](i: Int):
                var dequantized_lane = dequantized.load[width=simd_width](
                    group * self._group_size + i
                )
                var quantized_lane = round[DType.float32, simd_width](
                    dequantized_lane / scale_factor
                ).cast[DType.int8]()
                self._quantized.store[width=simd_width](
                    group * self._group_size + i, quantized_lane
                )

            vectorize[_quantize, nelts_f32](self._group_size)
        
        parallelize[quantize_group](num_groups, num_groups)
    
    fn rank(self) -> Int:
        return self._quantized.rank()
    
    fn quantize_naive(inout self, dequantized: TensorSlice[DType.float32]) raises:
        """Quantize the `dequantized` and update self. Uses a naive implementation.

        Args:
            dequantized: A tensor with float32 values to quantize.
        """

        if dequantized.shape() != self._quantized.shape():
            raise Error("shape mismatch, expected: " + str(self._quantized.shape()) + ", got: " + str(dequantized.shape()))

        var num_elements = dequantized.num_elements()

        if num_elements % self._group_size != 0:
            raise Error("number of elements must be a multiple of group size")

        var num_groups = num_elements // self._group_size

        var Q_MAX: Float32 = 127.0

        for group in range(num_groups):
            var wmax: Float32 = 0.0

            for i in range(self._group_size):
                var dequantized_lane = dequantized.load[width=1](
                    group * self._group_size + i
                )
                var local_wmax = abs(dequantized_lane)
                if local_wmax > wmax:
                    wmax = local_wmax

            var scale_factor = wmax / Q_MAX
            self._scale.store[width=1](group, scale_factor)

            # calculate and write back the quantized values
            for i in range(self._group_size):
                var dequantized_lane = dequantized.load[width=1](
                    group * self._group_size + i
                )
                var quantized_lane = round(dequantized_lane / scale_factor).cast[DType.int8]()
                self._quantized.store[width=1](
                    group * self._group_size + i, quantized_lane
                )
        
    fn fill(inout self, quantized: Int8, scale: Float32):
        """Fill the tensor with a constant value.

        Args:
            quantized: The quantized value to fill the tensor with.
            scale: The scale factor to fill the tensor with.
        """

        for i in range(self._quantized.num_elements()):
            self._quantized.store[width=1](i, quantized)
        
        for i in range(self._scale.num_elements()):
            self._scale.store[width=1](i, scale)


@value
struct QuantizedTensorSlice:
    """A reference to a quantized tensor."""

    var _quantized: DTypePointer[DType.int8]
    var _scale: DTypePointer[DType.float32]
    var _shape: TensorShape
    var _group_size: Int

    fn __init__(
        inout self,
        quantized: DTypePointer[DType.int8],
        scale: DTypePointer[DType.float32],
        shape: TensorShape,
        group_size: Int,
    ):
        self._quantized = quantized
        self._scale = scale
        self._shape = shape
        self._group_size = group_size

    fn __init__(inout self, qt: QuantizedTensor, layer: Int) raises:
        var num_layer_quantized_elements = qt._quantized.num_elements() / qt._quantized.dim(0)
        var num_layer_scale_elements = qt._scale.num_elements() / qt._scale.dim(0)

        self._quantized = qt._quantized.data().offset(layer * num_layer_quantized_elements)
        self._scale = qt._scale.data().offset(layer * num_layer_scale_elements)

        if qt._quantized.rank() == 2:
            self._shape = TensorShape(qt._quantized.dim(1))
        elif qt._quantized.rank() == 3:
            self._shape = TensorShape(qt._quantized.dim(1), qt._quantized.dim(2))
        else:
            raise Error("unimplemented rank")

        self._group_size = qt._group_size
    
    fn __init__(inout self, qt: QuantizedTensor, layer: Int, row: Int) raises:
        var num_layer_quantized_elements = qt._quantized.num_elements() / qt._quantized.dim(0)
        var num_layer_scale_elements = qt._scale.num_elements() / qt._scale.dim(0)
        var num_row_quantized_elements = num_layer_quantized_elements / qt._quantized.dim(1)
        var num_row_scale_elements = num_layer_scale_elements / qt._scale.dim(1)

        self._quantized = qt._quantized.data().offset(
            layer * num_layer_quantized_elements + row * num_row_quantized_elements
        )
        self._scale = qt._scale.data().offset(
            layer * num_layer_scale_elements + row * num_row_scale_elements
        )

        if qt._quantized.rank() == 3:
            self._shape = TensorShape(qt._quantized.dim(2))
        else:
            raise Error("unimplemented rank")

        self._group_size = qt._group_size


    fn __getitem__(self, idx: Int) -> SIMD[DType.int8, 1]:
        return self._quantized.load[width=1](idx)

    fn load[width: Int](self, idx: Int) -> SIMD[DType.int8, width]:
        return self._quantized.load[width=width](idx)

    fn store[width: Int](self, idx: Int, value: SIMD[DType.int8, width]):
        self._quantized.store[width=width](idx, value)

    fn quantized_data(self) -> DTypePointer[DType.int8]:
        return self._quantized

    fn scale_data(self) -> DTypePointer[DType.float32]:
        return self._scale

    fn shape(self) -> TensorShape:
        return self._shape

    fn num_elements(self) -> Int:
        return self._shape.num_elements()

    fn dim(self, idx: Int) -> Int:
        return self._shape[idx]

fn wrap(token: String) -> String:
    """Wrap special characters in the token.

    Args:
        token: The token to wrap.
    """
    alias a = String("\\n")
    alias b = String("\\t")
    alias c = String("'")
    alias d = String('"')
    if token == a:
        return String(List[Int8](0x0A, 0))
    if token == b:
        return String(List[Int8](0x09, 0))
    if token == c:
        return String(List[Int8](0x27, 0))
    if token == d:
        return String(List[Int8](0x22, 0))

    return token


fn string_from_bytes(owned bytes: List[Int8]) -> String:
    """Convert a list of bytes to a string.

    Args:
        bytes: The list of bytes to convert.
    """
    bytes.append(0)
    return bytes^


@value
struct Tokenizer:
    """A Byte Pair Encoding (BPE) tokenizer."""
    var vocab: List[String]
    var vocab_scores: List[Float32]
    var max_token_length: Int
    var vocab_size: Int
    var map_vocab_to_index: Dict[String, Int]

    fn __init__(inout self, vocab_size: Int, filename: String) raises:
        with open(filename, "rb") as f:

            @parameter
            fn read_bytes_as[dtype: DType](size: Int) raises -> SIMD[dtype, 1]:
                # a List that keeps ownership of the pointer
                var bytes = f.read_bytes(size)
                # copy one element of new type after casting pointer
                var result = bytes.data.bitcast[SIMD[dtype, 1]]()[0]
                # orginal List and data can be destroyed
                _ = bytes

                return result

            self.vocab_size = vocab_size
            self.vocab_scores = List[Float32](capacity=self.vocab_size)
            self.vocab = List[String](capacity=self.vocab_size)
            self.map_vocab_to_index = Dict[String, Int]()
            self.max_token_length = int(read_bytes_as[DType.int32](4))

            # read vocab_scores & vocab values (tokens)
            for i in range(self.vocab_size):
                var score = read_bytes_as[DType.float32](4)
                var slen = int(read_bytes_as[DType.int32](4))


                var token = string_from_bytes(f.read_bytes(slen))
                self.vocab.append(token^)
                self.vocab_scores.append(score)
                self.map_vocab_to_index[self.vocab[i]] = i

    fn find(self, token_o: String) -> Int:
        """Find the index of the token in the vocabulary.

        Args:
            token_o: The token to find.

        Returns:
            The index of the token in the vocabulary.
        """
        var token = wrap(token_o)
        var index = self.map_vocab_to_index.find(token)
        if index:
            return index.value()[]
        return -1


fn str_concat(s1: String, s2: String) -> String:
    """Concatenate two strings.

    Args:
        s1: The first string.
        s2: The second string.
    """
    var l1 = len(s1)
    var l2 = len(s2)
    var str = List[Int8](capacity=l1 + l2 + 1)
    memcpy(str.data, s1._buffer.data, l1)
    memcpy(str.data + l1, s2._buffer.data, l2)
    str[l1 + l2] = 0
    str.size = l1 + l2 + 1
    return str^


fn bpe_encode(inout tokens: List[Int], text: String, tok: Tokenizer):
    for pos in range(len(text)):
        var char = text[pos]
        var id = tok.find(char)
        if id == -1:
            print("Not a good prompt token at pos ", pos)
            return
        tokens.append(id)

    while True:
        var best_score = Float32(-1e10)
        var best_id = -1
        var best_idx = -1

        for i in range(len(tokens) - 1):
            # Check if we can merge the pair (tokens[i], tokens[i+1])
            # var str = str_concat(tok.vocab[tokens[i]], tok.vocab[tokens[i + 1]])
            # TODO: check: do we support add operator for string now?
            var str = tok.vocab[tokens[i]] + tok.vocab[tokens[i + 1]]
            var id = tok.find(str)
            if id != -1 and tok.vocab_scores[id] > best_score:
                best_score = tok.vocab_scores[id]
                best_id = id
                best_idx = i

        if best_idx == -1:
            # We couldn't find any more pairs to merge, so we're done
            break

        # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id
        # Delete token at position best_idx+1, shift the entire sequence back 1
        var _tokens = List[Int]()
        for i in range(0, best_idx + 1):
            _tokens.append(tokens[i])
        for i in range(best_idx + 2, len(tokens)):
            _tokens.append(tokens[i])
        tokens = _tokens^


fn read_bytes_as[dtype: DType](inout f: FileHandle) raises -> SIMD[dtype, 1]:
    var bytes = f.read_bytes(sizeof[dtype]())
    var result = bytes.data.bitcast[SIMD[dtype, 1]]()[0]
    _ = bytes
    return result

struct Config:
    """Configuration of the model."""
    var version: Int
    var dim: Int
    var kv_dim: Int
    var hidden_dim: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var kv_mul: Int
    var vocab_size: Int
    var seq_len: Int
    var head_size: Int
    var shared_weights: Bool
    var group_size: Int

    fn __init__(inout self, fileName: String, print_config: Bool) raises:
        #  Header (256 bytes)
        #  ------------------
        #  4 bytes:  Magic number ("ak42" - 0x616b3432)       // uint32
        #  4 bytes:  Version (2)                              // int32
        #  4 bytes:  Model dimension (p.dim)                  // int32
        #  4 bytes:  Hidden dimension (FF layer)              // int32
        #  4 bytes:  Number of layers (p.n_layers)            // int32
        #  4 bytes:  Number of attention heads (p.n_heads)    // int32
        #  4 bytes:  Number of key-value heads (n_kv_heads)   // int32
        #  4 bytes:  Vocabulary size (p.vocab_size)           // int32
        #  4 bytes:  Max sequence length (p.max_seq_len)      // int32
        #  1 byte:   Shared classifier flag                   // uint8
        #  4 bytes:  Group size for quantization              // int32
        #  Remaining: Zero padding to 256 bytes               // uint8[]
        #  
        #  FP32 Norm Parameters
        #  --------------------
        #  Attention norm weights (each layer)                // float32[]
        #  Feed-forward norm weights (each layer)             // float32[]
        #  Final norm weight before classifier                // float32[]
        #  
        #  Quantized Weights (Q8_0)
        #  ------------------------
        #  For each weight matrix:
        #  - Quantized int8 weights                           // int8[]
        #  - Scale factors (FP32)                             // float32[]
        with open(fileName, "rb") as f:
            var magic = read_bytes_as[DType.uint32](f)
            if magic != 0x616b3432:
                raise Error("Invalid magic number")
            self.version = int(read_bytes_as[DType.int32](f))
            self.dim = int(read_bytes_as[DType.int32](f))
            self.hidden_dim = int(read_bytes_as[DType.int32](f))
            self.n_layers = int(read_bytes_as[DType.int32](f))
            self.n_heads = int(read_bytes_as[DType.int32](f))
            self.n_kv_heads = int(read_bytes_as[DType.int32](f))
            self.vocab_size = int(read_bytes_as[DType.int32](f))
            self.seq_len = int(read_bytes_as[DType.int32](f))
            self.shared_weights = read_bytes_as[DType.uint8](f) == 1
            self.group_size = int(read_bytes_as[DType.int32](f))
            self.head_size = self.dim // self.n_heads
            self.kv_dim = (self.n_kv_heads * self.dim) // self.n_heads
            self.kv_mul = self.n_heads // self.n_kv_heads

        if print_config:
            print("config: dim, hidden_dim", self.dim, self.hidden_dim)
            print("config: n_layers, n_heads", self.n_layers, self.n_heads)
            print("config: vocab_size, seq_len", self.vocab_size, self.seq_len)
            print("config: head_size", self.head_size)
            print("config: kv_dim, kv_mul", self.kv_dim, self.kv_mul)
            print("config: shared_weights, group_size", self.shared_weights, self.group_size)

@value
struct RunState:
    var x: Tensor[DType.float32]  # activation at current time stamp (dim,)
    var xb: Tensor[DType.float32]  # same, but inside a residual branch (dim,)
    var xb2: Tensor[DType.float32]  # an additional buffer just for convenience (dim,)
    var hb: Tensor[DType.float32]  # buffer for hidden dimension in the ffn (hidden_dim,)
    var hb2: Tensor[DType.float32]  # buffer for hidden dimension in the ffn (hidden_dim,)
    var q: Tensor[DType.float32]  # query (dim,)
    var k: TensorSlice[DType.float32]  # key (kv_dim,)
    var v: TensorSlice[DType.float32]  # value (kv_dim,)
    var att: Tensor[DType.float32]  # buffer for scores/attention values (n_heads, seq_len)
    var logits: Tensor[DType.float32]  # output logits
    var key_cache: Tensor[DType.float32]  # (layer, seq_len, dim)
    var value_cache: Tensor[DType.float32]  # (layer, seq_len, dim)
    var x_q: QuantizedTensor # quantized x (dim,)
    var hb_q: QuantizedTensor # quantized hb (hidden_dim,)

    fn __init__(inout self, config: Config) raises:
        self.x = Tensor[DType.float32](config.dim)
        self.xb = Tensor[DType.float32](config.dim)
        self.xb2 = Tensor[DType.float32](config.dim)
        self.hb = Tensor[DType.float32](config.hidden_dim)
        self.hb2 = Tensor[DType.float32](config.hidden_dim)
        self.q = Tensor[DType.float32](config.dim)
        self.att = Tensor[DType.float32](TensorShape((config.n_heads, config.seq_len)))
        self.logits = Tensor[DType.float32](config.vocab_size)
        self.key_cache = Tensor[DType.float32](TensorShape((
            config.n_layers, config.seq_len, config.kv_dim
        )))
        self.value_cache = Tensor[DType.float32](TensorShape((
            config.n_layers, config.seq_len, config.kv_dim
        )))
        # So their updates flow to the caches, k and v are slices with shared memory.
        # Initialize with placeholders. The real tensors reference layer and position during forward pass.
        self.k = TensorSlice(Tensor[DType.float32](TensorShape((1, config.kv_dim))), 1)
        self.v = TensorSlice(Tensor[DType.float32](TensorShape((1, config.kv_dim))), 1)
        self.x_q = QuantizedTensor(
            TensorShape(config.dim),
            int(config.group_size)
        )
        self.hb_q = QuantizedTensor(
            TensorShape(config.hidden_dim),
            int(config.group_size)
        )

@value
struct TransformerWeights:
    var token_embedding_table: Tensor[DType.float32]
    var q_token_embedding_table: QuantizedTensor

    var rms_att_weight: Tensor[DType.float32]
    var rms_ffn_weight: Tensor[DType.float32]

    # var freq_cis_real: TensorF32
    # var freq_cis_imag: TensorF32
    
    var wq: QuantizedTensor
    var wk: QuantizedTensor
    var wv: QuantizedTensor
    var wo: QuantizedTensor

    var w1: QuantizedTensor
    var w2: QuantizedTensor
    var w3: QuantizedTensor

    var rms_final_weight: Tensor[DType.float32]
    var wcls: QuantizedTensor

    fn __init__(inout self, file_name: String, config: Config) raises:
        var bytes_read = 0
        with open(file_name, "rb") as f:
            
            @parameter
            fn read_weights_fp32(shape: TensorShape) raises -> Tensor[DType.float32]:
                var bytes = f.read_bytes(shape.num_elements() * sizeof[DType.float32]())
                if bytes.size != shape.num_elements() * sizeof[DType.float32]():
                    raise Error("EOF while reading weights")
                bytes_read += bytes.size
                var data = bytes.steal_data().bitcast[Float32]()
                
                return Tensor[DType.float32](shape, data)
            
            @parameter
            fn read_weights_i8(shape: TensorShape) raises -> QuantizedTensor:
                """Read quantized weights from the file.
                
                Args:
                    shape: The shape of the weights, should be (layer, dim, ...)
                """
                if shape.rank() <= 1:
                    raise Error("invalid shape for quantized weights, should be (layer, dim, ...)")

                var tensor = QuantizedTensor(shape, config.group_size)

                var n_layers = shape[0]
                for i in range(n_layers):
                    var tensor_layer = QuantizedTensorSlice(tensor, i)

                    # read int8 weights
                    var weight_bytes = f.read_bytes(shape.num_elements() // n_layers * sizeof[DType.int8]())
                    if weight_bytes.size != shape.num_elements() // n_layers * sizeof[DType.int8]():
                        raise Error("EOF while reading weights")
                    var weight_size = weight_bytes.size
                    bytes_read += weight_size
                    var weight_data = weight_bytes.steal_data()
                    
                    # read scale factors
                    var scale_bytes = f.read_bytes(shape.num_elements() // n_layers // config.group_size * sizeof[DType.float32]())
                    if scale_bytes.size != shape.num_elements() // n_layers // config.group_size * sizeof[DType.float32]():
                        raise Error("EOF while reading scale factors")
                    var scale_size = scale_bytes.size
                    bytes_read += scale_size
                    var scale_data = scale_bytes.steal_data().bitcast[Float32]()

                    memcpy(tensor_layer.quantized_data(), weight_data, weight_size)
                    memcpy(tensor_layer.scale_data(), scale_data, scale_size // sizeof[DType.float32]())

                    # not sure if we need to free the data here
                    weight_data.free()
                    scale_data.free()
                
                return tensor
            
            # 256 bytes for the config header
            var config_header_bytes = f.read_bytes(NUM_CONFIG_HEADER_BYTES)
            bytes_read += config_header_bytes.size
            print("header done, bytes read:", bytes_read)

            # rms_att_weight
            self.rms_att_weight = read_weights_fp32(TensorShape(config.n_layers, config.dim))
            print("rms_att_weight done, bytes read:", bytes_read)

            # rms_ffn_weight
            self.rms_ffn_weight = read_weights_fp32(TensorShape(config.n_layers, config.dim))
            print("rms_ffn_weight done, bytes read:", bytes_read)

            # rms_final_weight
            self.rms_final_weight = read_weights_fp32(TensorShape(config.dim))
            print("rms_final_weight done, bytes read:", bytes_read)

            # q_token_embedding_table
            self.q_token_embedding_table = read_weights_i8(TensorShape(1, config.vocab_size, config.dim)) # expand layer dim = 1
            print("q_token_embedding_table done, bytes read:", bytes_read)

            # dequantize token_embedding_table
            self.token_embedding_table = Tensor[DType.float32](TensorShape(config.vocab_size, config.dim))
            print("token_embedding_table done, bytes read:", bytes_read)
            self.q_token_embedding_table.dequantize(TensorSlice(self.token_embedding_table.data(), TensorShape(config.vocab_size, config.dim)))
            print("dequantize token_embedding_table done, bytes read:", bytes_read)

            # wq, wk, wv, wo
            self.wq = read_weights_i8(TensorShape(config.n_layers, config.dim, config.dim))
            self.wk = read_weights_i8(TensorShape(config.n_layers, config.kv_dim, config.dim))
            self.wv = read_weights_i8(TensorShape(config.n_layers, config.kv_dim, config.dim))
            self.wo = read_weights_i8(TensorShape(config.n_layers, config.dim, config.dim))
            print("wq, wk, wv, wo done, bytes read:", bytes_read)

            # w1, w2, w3
            self.w1 = read_weights_i8(TensorShape(config.n_layers, config.hidden_dim, config.dim))
            self.w2 = read_weights_i8(TensorShape(config.n_layers, config.dim, config.hidden_dim))
            self.w3 = read_weights_i8(TensorShape(config.n_layers, config.hidden_dim, config.dim))
            print("w1, w2, w3 done, bytes read:", bytes_read)

            # wcls
            if config.shared_weights:
                self.wcls = self.wq
            else:
                self.wcls = read_weights_i8(TensorShape(1, config.vocab_size, config.dim))
            print("wcls done, bytes read:", bytes_read)
            
# From: llama2.mojo
@register_passable
struct Accumulator[T: DType, width: Int]:
    # ideally this could be SIMD[T, width] but the width
    # in accumulate() method is compared by identity
    var data: DTypePointer[T]

    @always_inline
    fn __init__() -> Self:
        # allocate a DTypePointer on stack that doesn't need to be freed.
        var data = stack_allocation[width, T]()
        memset_zero(data, width)
        return Self {data: data}

    @always_inline
    fn accumulate[_width: Int](inout self, val: SIMD[T, _width]) -> None:
        # This is a hack to make sure both SIMD have _width length.
        # SIMD[T, width] += SIMD[T, _width] is always an error.
        var newVal = self.data.load[width=_width]() + val
        self.data.store[width=_width](newVal)

    @always_inline
    fn total(self) -> SIMD[T, 1]:
        return self.data.load[width=width]().reduce_add()

@always_inline
fn rmsnorm(
    inout o: DTypePointer[DType.float32],
    x: DTypePointer[DType.float32],
    weight: DTypePointer[DType.float32],
    size: Int,
) -> None:
    # Calculate sum of squares
    var tmp = Accumulator[DType.float32, nelts_f32]()

    @parameter
    fn _sum2[_nelts: Int](j: Int):
        tmp.accumulate(x.offset(j).load[width=_nelts](0) ** 2)

    vectorize[_sum2, nelts_f32](size)

    var ss: Float32 = tmp.total()
    ss = ss / size + 1e-5
    ss = 1.0 / math.sqrt(ss)

    # Normalize and scale
    @parameter
    fn _norm[_nelts: Int](j: Int):
        var val = weight.load[width=_nelts](j) * ss * x.load[width=_nelts](j)
        o.offset(j).store[width=_nelts](0, val)

    vectorize[_norm, nelts_f32](size)


@always_inline
fn rmsnorm(inout o: Tensor[DType.float32], x: Tensor[DType.float32], weight: Tensor[DType.float32]):
    rmsnorm(o._ptr, x.data(), weight.data(), weight.dim(weight.rank() - 1))


@always_inline
fn rmsnorm(inout o: Tensor[DType.float32], x: Tensor[DType.float32], weight: TensorSlice[DType.float32]):
    rmsnorm(o._ptr, x.data(), weight.data(), weight.dim(weight.rank() - 1))

@always_inline
fn softmax(inout x: Tensor[DType.float32]) -> None:
    softmax(x, 0, x.dim(0))


@always_inline
fn softmax(inout x: Tensor[DType.float32], start: Int, end: Int):
    var max_val: Float32 = -1e9

    @parameter
    fn _max[_nelts: Int](ii: Int):
        var val = x.load[width=_nelts](start + ii).reduce_max()
        if val > max_val:
            max_val = val

    vectorize[_max, nelts_f32](end - start)

    var acc = Accumulator[DType.float32, nelts_f32]()

    @parameter
    fn _exp[_nelts: Int](ii: Int):
        var val = math.exp(x.load[width=_nelts](start + ii) - max_val)
        x.store[width=_nelts](start + ii, val)
        acc.accumulate(val)

    vectorize[_exp, nelts_f32](end - start)

    var ssum = acc.total()

    @parameter
    fn _norm[_nelts: Int](ii: Int):
        x.store[width=_nelts](
            start + ii, x.load[width=_nelts](start + ii) / ssum
        )

    vectorize[_norm, nelts_f32](end - start)

@always_inline
fn batch_matmul_fp32[
    n: Int
](
    C: StaticTuple[DTypePointer[DType.float32], n],
    A: DTypePointer[DType.float32],
    B: StaticTuple[DTypePointer[DType.float32], n],
    rows: Int,
    cols: Int,
):
    @parameter
    fn compute_row(i: Int):
        var tmp = StaticTuple[Accumulator[DType.float32, nelts_f32], n]()

        @unroll
        for k in range(n):
            tmp[k] = Accumulator[DType.float32, nelts_f32]()

        var row_offset = i * cols

        @parameter
        fn dot[_nelts: Int](j: Int):
            var a = A.load[width=_nelts](j)

            @unroll
            for k in range(n):
                tmp[k].accumulate(a * B[k].load[width=_nelts](row_offset + j))

        vectorize[dot, nelts_f32](cols)

        @unroll
        for k in range(n):
            C[k].store(i, tmp[k].total())

    parallelize[compute_row](rows, workers)

@always_inline
fn batch_matmul_i8[
    n: Int
](
    C: StaticTuple[DTypePointer[DType.float32], n],
    A: QuantizedTensor,
    B: StaticTuple[UnsafePointer[QuantizedTensorSlice], n],
    rows: Int,
    cols: Int,
):
    # B (rows, cols) @ A (cols) = C (rows)
    # print("A shape: ", A._quantized.shape(), A._scale.shape())
    var group_size = A._group_size

    @parameter
    fn compute_row(i: Int):
        var val = StaticTuple[Float32, n](0)
        var offset = i * cols
        for j in range(cols // group_size):
            var ival = StaticTuple[Accumulator[DType.int32, nelts_q32], n]()
            
            @unroll
            for k in range(n):
                ival[k] = Accumulator[DType.int32, nelts_q32]()

            @parameter
            fn dot[_nelts: Int](k: Int):
                @unroll
                for idx in range(n):
                    ival[idx].accumulate(A._quantized.load[width=_nelts](k + j * group_size).cast[DType.int32]()
                        * B[idx][]._quantized.load[width=_nelts](k + offset + j * group_size).cast[DType.int32]())
            
            vectorize[dot, nelts_q32](group_size)

            @unroll
            for idx in range(n):
                val[idx] += ival[idx].total().cast[DType.float32]() * A._scale.load[width=1](j) * B[idx][]._scale.load[width=1](offset // group_size + j)
        
        @unroll
        for idx in range(n):
            C[idx].store(i, val[idx])
        
    parallelize[compute_row](rows, workers)


def test():
    def test_quantized_tensor():
        alias N = 4096
        alias group_size = 512
        print("QuantizedTensor test")
        var qt = QuantizedTensor(
            TensorShape(N),
            group_size
        )
        var qt_naive = QuantizedTensor(
            TensorShape(N),
            group_size
        )

        var x = rand[DType.float32](5, N) - 0.5

        var x_1 = TensorSlice[DType.float32](x, 0)
        var x_1_naive = TensorSlice[DType.float32](x, 0)
        # print("Original:")
        # print(x_1.load[width=N](0))

        qt.quantize(x_1)
        qt_naive.quantize_naive(x_1_naive)

        # print("Quantized:")
        # print(qt._quantized.load[width=N](0))

        # print("Quantized (naive):")
        # print(qt_naive._quantized.load[width=N](0))

        # print("Scale factors:")
        # print(qt._scale.load[width=N // group_size](0))

        # print("Scale factors (naive):")
        # print(qt_naive._scale.load[width=N // group_size](0))

        var x_2 = TensorSlice[DType.float32](x, 3)
        var x_2_naive = TensorSlice[DType.float32](x, 4)

        qt.dequantize(x_2)
        qt_naive.dequantize_naive(x_2_naive)

        # print("Dequantized:")
        # print(x_2.load[width=N](0))

        # print("Dequantized (naive):")
        # print(x_2_naive.load[width=N](0))

        var error = x_1.load[width=N](0) - x_2.load[width=N](0)
        var error_naive = x_1_naive.load[width=N](0) - x_2_naive.load[width=N](0)

        # print("Error:")
        # print(error)

        # print("Error (naive):")
        # print(error_naive)

        var max_error = abs(error).reduce_max()
        print("Max error:", max_error)

        var max_error_naive = abs(error_naive).reduce_max()
        print("Max error (naive):", max_error_naive)

        if max_error != max_error_naive:
            print("Error: max_error != max_error_naive")
            return

        if max_error > 1e-2:
            print("Error: max_error > 1e-2")
            return

        print("QuantizedTensor test passed")

        _ = x


    def test_bpe_encode():
        print("BPE encode test")

        var name = StringRef("tokenizer.bin")
        var tok = Tokenizer(128256, name)
        var prompt_tokens = List[Int]()
        var gt = SIMD[DType.int32, 8](9906, 11, 1268, 527, 499, 30)

        bpe_encode(prompt_tokens, "Hello, how are you?", tok)

        for i in range(len(prompt_tokens)):
            if(prompt_tokens[i] != int(gt[i])):
                print("Error at index ", i)
                print("Expected: ", gt[i])
                print("Got: ", prompt_tokens[i])
        
        print("BPE encode test passed")

    
    def test_transformer_weights():
        print("Transformer weights test")
        var config = Config("llama3_8b_instruct_q80.bin", True)
        var weights = TransformerWeights("llama3_8b_instruct_q80.bin", config)
        print("Transformer weights test passed")
    
    def test_batch_matmul_i8():
        alias dim = 4096
        alias kv_dim = 1024
        alias group_size = 64
        alias n_layers = 32
        var x_q = QuantizedTensor(TensorShape(dim), group_size)
        x_q.fill(1, 0.1)

        var wk = QuantizedTensor(TensorShape(n_layers, kv_dim, dim), group_size)
        var wv = QuantizedTensor(TensorShape(n_layers, kv_dim, dim), group_size)
        wk.fill(1, 0.1)
        wv.fill(1, 0.1)

        var layer = 0
        var wk_slice = QuantizedTensorSlice(wk, layer)
        var wv_slice = QuantizedTensorSlice(wv, layer)

        var k = Tensor[DType.float32](TensorShape(kv_dim))
        var v = Tensor[DType.float32](TensorShape(kv_dim))

        batch_matmul_i8(
            StaticTuple[DTypePointer[DType.float32], 2](k.data(), v.data()), 
            x_q, 
            StaticTuple[UnsafePointer[QuantizedTensorSlice], 2](
                UnsafePointer(wk_slice), 
                UnsafePointer(wv_slice)
            ), 
            kv_dim, 
            dim)

        print("Quantized batch matmul test")
        print("x_q:", x_q._quantized.data()[0], " ", x_q._scale.data()[0])
        print("wk:", wk._quantized.data()[0], " ", wk._scale.data()[0])
        print("wv:", wv._quantized.data()[0], " ", wv._scale.data()[0])
        print("k:", k.data()[0])
        print("v:", v.data()[0])

        if k.data()[0] - 40.96 > 1e-2:
            print("Error: k.data()[0] != 64")
            return
        
        if v.data()[0] - 40.96 > 1e-2:
            print("Error: v.data()[0] != 64")
            return

        _ = x_q
        _ = wk
        _ = wv
        _ = k
        _ = v

        print("Quantized batch matmul test passed")


    # test_quantized_tensor()
    # test_bpe_encode()
    # test_transformer_weights()
    test_batch_matmul_i8()

def main():
    workers = num_performance_cores()
    test()
    
