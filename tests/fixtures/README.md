# tests/fixtures

Verbatim third-party files used as test inputs.

`convert_hf_to_gguf_master.py.txt` is the llama.cpp `convert_hf_to_gguf.py`
entrypoint as fetched from
`https://github.com/ggerganov/llama.cpp/raw/refs/heads/master/convert_hf_to_gguf.py`,
the exact URL `unsloth_zoo/llama_cpp.py` downloads at export time. It is the
"a genuine converter raises nothing" input for
`tests/test_llama_cpp_converter_scan.py`.

  * sha256 `e9a1da876330bbce9687541ab31736542a01b4ac43c6686126514a50f122fb7f`
  * 13012 bytes, the post-split entrypoint that imports the `conversion` package

The `.py.txt` suffix is deliberate: the bytes are upstream's, not ours, and the
repo's tree-wide lint gates walk every `.py` file. Refresh it by re-downloading
from that URL and updating the hash and size above.
