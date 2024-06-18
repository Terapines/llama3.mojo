# llama3.mojo

llama3 in mojo🔥

**Still work in progress!**

This repo is based on [llama2.mojo](https://github.com/tairov/llama2.mojo) and [llama3.c](https://github.com/jameswdelancey/llama3.c), with support for llama3-8b inference in pure mojo.

Currently llama3.mojo only supports llama3-8b inference with Q_8 quantization. 70B and fp32 support are on the way.

## Getting started

First, Install [Mojo🔥](https://docs.modular.com/mojo/manual/get-started) 

Due to llama's license, the model and tokenizer are not included in the repository, you need to download them following [Meta's instructions](https://github.com/meta-llama/llama3) 

Then export and quantize Llama3-8B using code from [llama3.c](https://github.com/jameswdelancey/llama3.c)

See https://github.com/jameswdelancey/llama3.c?tab=readme-ov-file#metas-llama-3-models

and https://github.com/jameswdelancey/llama3.c?tab=readme-ov-file#int8-quantization

You will get a model file like `llama3_8b_instruct_q80.bin` and a tokenizer file like `tokenizer.bin`

## Run inference

```bash
mojo llama3q.mojo llama3_8b_instruct_q80.bin -z tokenizer.bin -i "The planets of the solar system are" -n 128
```

Note that loading model will take some time since the model is large, and memcpy is used.

The output will look like this:

```
num parallel workers: 6  SIMD width: float32: 32  int32: 32  int8: 128
Reading weights...
header done, bytes read: 256
rms_att_weight done, bytes read: 524544
rms_ffn_weight done, bytes read: 1048832
rms_final_weight done, bytes read: 1065216
q_token_embedding_table done, bytes read: 559235328
token_embedding_table done, bytes read: 559235328
dequantize token_embedding_table done, bytes read: 559235328
wq, wk, wv, wo done, bytes read: 1985298688
w1, w2, w3 done, bytes read: 7974764800
wcls done, bytes read: 8532934912
n layers: 32 | vocab size: 128256
The planets of the solar system are listed in order of their average distance from the sun, starting with Mercury and moving outward to Neptune. The average distance of each planet from the sun is known as its semi-major axis.
The planets of the solar system, listed in order of their average distance from the sun, are:
1. Mercury - 58 million kilometers (36 million miles)
2. Venus - 108 million kilometers (67 million miles)
3. Earth - 149.6 million kilometers (92.96 million miles)
4. Mars - 227.9 million kilometers (141.6 million miles)
5. Jupiter -
```

## Future work
- [ ] Performance optimization (RoPE, Quantization, Matmul, etc)
- [ ] Performance benchmark
- [ ] Support for 70B
- [ ] Support for fp32

## Reference

This repo is largely based on [llama2.mojo](https://github.com/tairov/llama2.mojo) and [llama3.c](https://github.com/jameswdelancey/llama3.c)

[llama3 from Meta](https://github.com/meta-llama/llama3)




