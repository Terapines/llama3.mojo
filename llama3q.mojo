from tensor import Tensor, TensorShape, rand
from algorithm import vectorize, parallelize
from math import abs, round

alias nelts_q8 = (4 * simdwidthof[Int8]())
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
    var version: Int32
    var dim: Int32
    var kv_dim: Int32
    var hidden_dim: Int32
    var n_layers: Int32
    var n_heads: Int32
    var n_kv_heads: Int32
    var kv_mul: Int32
    var vocab_size: Int32
    var seq_len: Int32
    var head_size: Int32
    var shared_weights: Bool
    var group_size: Int32

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
            self.version = read_bytes_as[DType.int32](f)
            self.dim = read_bytes_as[DType.int32](f)
            self.hidden_dim = read_bytes_as[DType.int32](f)
            self.n_layers = read_bytes_as[DType.int32](f)
            self.n_heads = read_bytes_as[DType.int32](f)
            self.n_kv_heads = read_bytes_as[DType.int32](f)
            self.vocab_size = read_bytes_as[DType.int32](f)
            self.seq_len = read_bytes_as[DType.int32](f)
            self.shared_weights = read_bytes_as[DType.uint8](f) == 1
            self.group_size = read_bytes_as[DType.int32](f)
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

def test():
    def test_quantized_tensor():
        alias N = 4096
        alias group_size = 512
        print("QuantizedTensor test")
        var qt = QuantizedTensor(
            TensorShape(N),
            int(group_size)
        )
        var qt_naive = QuantizedTensor(
            TensorShape(N),
            int(group_size)
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
    
    test_quantized_tensor()
    test_bpe_encode()

def main():
    test()
    
