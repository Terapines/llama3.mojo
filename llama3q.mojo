from tensor import Tensor, TensorShape
from algorithm import vectorize

alias nelts_u8 = (4 * simdwidthof[UInt8]())
alias nelts_f32 = (4 * simdwidthof[Float32]())


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
    """An 8-bit quantized tensor.

    """

    var _quantized: TensorSlice[DType.uint8]
    var _scale: TensorSlice[DType.float32]
    var _group_size: Int

    fn __init__(
        inout self,
        ptr: DTypePointer[DType.uint8],
        shape: TensorShape,
        scale_ptr: DTypePointer[DType.float32],
        group_size: Int,
    ):
        self._quantized = TensorSlice[DType.uint8](ptr, shape)
        var num_scale_factors = self._quantized.num_elements() // group_size
        self._scale = TensorSlice[DType.float32](scale_ptr, TensorShape(num_scale_factors))
        self._group_size = group_size

    fn dequantize(self, dequantized: TensorSlice[DType.float32]):
        """Dequantize the tensor into `dequantized`.

        Args:
            dequantized: The tensor to store the dequantized values.
        """
        var num_elements = self._quantized.num_elements()

        @parameter
        fn _dequantize[simd_width: Int](i: Int):
            var group = i // self._group_size
            var scale_factor = self._scale[group]

            var quantized_lane = self._quantized.load[width=simd_width](i).cast[DType.float32]()
            dequantized.store[width=simd_width](i, quantized_lane * scale_factor)

        vectorize[_dequantize, nelts_u8](num_elements)

    fn quantize(inout self, dequantized: TensorSlice[DType.float32]) raises:
        """Quantize the `dequantized` and update self.

        Args:
            dequantized: A tensor with float32 values to quantize.
        """

        var num_elements = dequantized.num_elements()
        
        if num_elements % self._group_size != 0:
            raise Error("number of elements must be a multiple of group size")

        # TODO: Implement quantization

        pass


fn wrap(token: String) -> String:
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
    bytes.append(0)
    return bytes^


@value
struct Tokenizer:
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

                # print(slen)

                var token = string_from_bytes(f.read_bytes(slen))
                self.vocab.append(token^)
                self.vocab_scores.append(score)
                self.map_vocab_to_index[self.vocab[i]] = i

    fn find(self, token_o: String) -> Int:
        var token = wrap(token_o)
        var index = self.map_vocab_to_index.find(token)
        if index:
            return index.value()[]
        return -1

fn str_concat(s1: String, s2: String) -> String:
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
            var str = str_concat(tok.vocab[tokens[i]], tok.vocab[tokens[i + 1]])
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
        
def main():
    var name = StringRef("tokenizer.bin")
    var tok = Tokenizer(128256, name)
    var prompt_tokens = List[Int]()

    bpe_encode(prompt_tokens, "Hello, how are you?", tok)

    for i in range(len(prompt_tokens)):
        print(prompt_tokens[i])