# TeLLMe: An Efficient End-to-End Ternary LLM Prefill and Decode Accelerator with Table-Lookup Matmul on Edge FPGAs [FPGA2026]

## Directory Structure

```
release/
├── on_board_test/       # On-board testing and chatbot runtime
│   ├── src/
│   │   ├── chatbot.py           # Main chatbot application
│   │   └── neon_lm_head/        # NEON-optimized LM head
│   ├── tokenizer/               # Tokenizer implementation
│   ├── weight_bin_5/            # Model weights
│   ├── kv_cache/                # KV cache data
│   ├── rope_cache/              # RoPE cache
│   ├── input_bin/               # Input embeddings
│   ├── output_bin/              # Output data (HLS golden)
│   ├── output_bin_pytorch/      # Output data (PyTorch golden)
│   └── scripts/                 # Helper scripts
│
└── src/                         # HLS C/C++ source code
    ├── config/
    │   ├── config.hpp           # Hardware configuration
    │   ├── macro.hpp            # Synthesis macros
    │   └── w_quant_*.txt        # Quantization tables
    ├── top_bitnet.hpp           # Top-level function
    ├── linear.hpp               # TMAC operations
    ├── load_weight.hpp          # Weight loading
    ├── reverse_attention.hpp    # Attention mechanism
    └── [other modules]
```

---

## On-Board Chatbot Evaluation

### Environment Setup

**Required Hardware:**
- KV260 FPGA board with PYNQ setup

**Required Software:**
- PYNQ environment

### Setup Instructions

#### 1. Download Required Data Files

**Important:** The release package does not include large binary data files (weights, cache data, etc.) due to size constraints.

**Download all required binary files from Google Drive:**

[Download Data Files Here](https://drive.google.com/drive/folders/1jrOIA7jaPUJkjDhWwoLr_jXlYaViF0vz?usp=sharing)

**Required directories and files:**
- `input_bin/` - Input embedding data
- `weight_bin_5/` - Model weights (1.58-bit ternary quantized)
- `kv_cache/` - KV cache data (sample_1 to sample_15 folders)
- `rope_cache/` - RoPE cache data
- `output_bin/` - HLS golden output (for verification)
- `output_bin_pytorch/` - PyTorch golden output (for verification with `--use_pytorch_golden`)

**Installation steps:**
1. Visit the Google Drive link above
2. Download the entire `on_board_test/` folder structure
3. Extract downloaded files to match the directory structure shown in "Directory Structure" section
4. Ensure all binary files (.bin, .dat) are in their respective directories

**Tip:** Each data directory contains a `DOWNLOAD_DATA.txt` file with specific instructions.

#### 2. Transfer Files to FPGA Board
Copy the entire `on_board_test/` directory (including downloaded data files) to your KV260 board.

#### 3. Switch to Root and Load PYNQ Environment
```bash
sudo su
cd path_to_on_board_test_directory/src
source /etc/profile.d/pynq_venv.sh
```

#### 4. Compile NEON LM Head (Required)
```bash
cd path_to_on_board_test_directory/src/neon_lm_head
make pybind
```
This compiles the ARM NEON-accelerated matrix multiplication library used by the `--neon_lm_head` flag.

---

### Running the Chatbot

#### Verification Mode (Golden Data Testing)

Test with golden data from PyTorch in **prefill** and **decode** modes:

**Prefill Mode:**
```bash
python3 chatbot.py --verify_golden --verify_mode prefill --sample_id "1-15" \
  --neon_lm_head --bitstream bitnet1024tl_250_dram_divide5.bit --use_pytorch_golden
```

**Decode Mode:**
```bash
python3 chatbot.py --verify_golden --verify_mode decode --sample_id "1-15" \
  --neon_lm_head --bitstream bitnet1024tl_250_dram_divide5.bit --use_pytorch_golden
```

Results will be saved to `on_board_prefill.log` and `on_board_decode.log`.

#### Interactive Chatbot Mode

Run an interactive chatbot session:
```bash
python3 chatbot.py --interactive --bitstream bitnet1024tl_250_dram_divide5.bit \
  --sampling topk --top_k 3 --neon_lm_head --stream --max_length 1023
```

**Note:** The model responses may vary in quality as the base model is not chat-finetuned.

```bash
taskset -c 0-3 chrt -f 99 
```
This command is also recommended to ensure the chatbot process has real-time scheduling priority for better performance.

---

### Command-Line Arguments Reference

| Argument | Type | Default | Description | Recommended |
|----------|------|---------|-------------|-------------|
| `--prompt` | str | "Hello, how are you?" | Input prompt for generation | - |
| `--max_length` | int | 256 | Maximum number of new tokens to generate | - |
| `--bitstream` | str | `bitnet1024tl_250_dram_divide5.bit` | Path to FPGA bitstream file | ✓ (use default) |
| `--no_fpga` | flag | False | Run in CPU simulation mode (no FPGA) | - |
| `--interactive` | flag | False | Run in interactive chat mode | ✓ (for chat) |
| `--verify_golden` | flag | False | Run verification using golden data | ✓ (for testing) |
| `--verify_mode` | str | "prefill" | Verification mode: `prefill` or `decode` | - |
| `--sample_id` | str | "0" | Sample ID(s): single ('5'), comma-separated ('0,1,2'), or range ('1-15') | ✓ (use "1-15") |
| `--sampling` | str | "argmax" | Sampling mode: `argmax`, `topk`, `topk_uniform` | - |
| `--top_k` | int | 16 | K value for top-k sampling | - |
| `--neon_lm_head` | flag | False | Use NEON-accelerated LM head | ✓ **Always** |
| `--stream` | flag | False | Enable streaming output | ✓ (for interactive) |
| `--use_pytorch_golden` | flag | False | Use PyTorch golden logits (output_bin_pytorch) | ✓ (for verification) |

---

## HLS Synthesis & Implementation

The `src/` directory contains production-ready HLS C/C++ source code for FPGA synthesis.

**Top Function:** `top_bitnet_fuse_ele_multi_weight_port()` in `src/top_bitnet.hpp`

You can synthesize and implement the design using:
- **Script-based flow:** Use Vitis HLS scripts for automated synthesis
- **GUI-based flow:** Import source files into Vitis HLS GUI project

The synthesized IP can then be integrated into Vivado for bitstream generation.

---

## License

This project is licensed under the **BSD 3-Clause License**. See the license headers in individual source files for details.

---

## Contributors

**Paper Contributors** (in order of authorship):
- Ye Qiao* and Zhiheng Chen* - University of California, Irvine  
- Yifan Zhang - University of California, Irvine
- Yian Wang - University of California, Irvine
- Sitao Huang - University of California, Irvine

*Co-first authors



---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{qiao2025tellme,
  title={TeLLMe: An Efficient End-to-End Ternary LLM Prefill and Decode Accelerator with Table-Lookup Matmul on Edge FPGAs},
  author={Qiao, Ye and Chen, Zhiheng and Zhang, Yifan and Wang, Yian and Huang, Sitao},
  booktitle={Proceedings of the 2026 ACM/SIGDA International Symposium on Field Programmable Gate Arrays},
  pages={247--257},
  year={2026},
  publisher={ACM},
  doi={10.1145/3748173.3779191},
  url={https://dl.acm.org/doi/10.1145/3748173.3779191}
}

@article{qiao2024tellme,
  title={TeLLMe v2: An Efficient End-to-End Ternary LLM Prefill and Decode Accelerator with Table-Lookup Matmul on Edge FPGAs},
  author={Qiao, Ye and Chen, Zhiheng and Zhang, Yifan and Wang, Yian and Huang, Sitao},
  journal={arXiv preprint arXiv:2510.15926},
  year={2024},
  url={https://arxiv.org/abs/2510.15926}
}

@article{qiao2025tellme_energy,
  title={TeLLMe: An Energy-Efficient Ternary LLM Accelerator for Prefilling and Decoding on Edge FPGAs},
  author={Qiao, Ye and Chen, Zhiheng and Zhang, Yifan and Wang, Yian and Huang, Sitao},
  journal={arXiv preprint arXiv:2504.16266},
  year={2025},
  url={https://arxiv.org/abs/2504.16266}
}
```

---

## References
- [BitNet b1.58 Paper](https://arxiv.org/abs/2402.17764)
- [BitNet b1.58 Model on Hugging Face](https://huggingface.co/1bitLLM/bitnet_b1_58-large/)



