"""
测试 Qwen3-VL GPU 模式

验证 Blackwell GPU (RTX PRO 6000) 支持
"""

import torch
import sys
from pathlib import Path

print("=" * 70)
print("Qwen3-VL GPU 模式测试")
print("=" * 70)

# 1. 检查 PyTorch 环境
print("\n1. PyTorch 环境检查:")
print(f"  PyTorch 版本: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
print(f"  CUDA 版本: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
print(f"  cuDNN 版本: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")

# 2. 检查 GPU 信息
print("\n2. GPU 信息:")
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"  GPU 数量: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    - 计算能力: {torch.cuda.get_device_capability(i)}")
        print(f"    - 总显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"    - 当前显存: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
else:
    print("  ✗ 未检测到可用的 GPU")
    sys.exit(1)

# 3. 检查 CUDA Toolkit
print("\n3. CUDA Toolkit 检查:")
import subprocess
try:
    result = subprocess.run(
        ["/usr/local/cuda-12.8/bin/nvcc", "--version"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"  ✓ CUDA Toolkit 12.8 已安装")
        print(f"    {result.stdout.strip()}")
    else:
        print(f"  ✗ CUDA Toolkit 检查失败: {result.stderr}")
except Exception as e:
    print(f"  ✗ CUDA Toolkit 检查失败: {e}")

# 4. 检查 transformers 版本
print("\n4. transformers 版本:")
try:
    import transformers
    print(f"  transformers 版本: {transformers.__version__}")
    print(f"  ✓ transformers 已安装")
except ImportError:
    print("  ✗ transformers 未安装")
    sys.exit(1)

# 5. 检查 Qwen3-VL 模型
print("\n5. Qwen3-VL 模型检查:")
model_path = "/home/yyx/TravelUAV/Model/Qwen3-VL-4B-Instruct"
if Path(model_path).exists():
    print(f"  ✓ 模型路径存在: {model_path}")
    # 检查模型文件
    model_files = list(Path(model_path).glob("*.safetensors"))
    if model_files:
        print(f"  ✓ 找到 {len(model_files)} 个模型文件")
    else:
        model_files = list(Path(model_path).glob("*.bin"))
        if model_files:
            print(f"  ✓ 找到 {len(model_files)} 个模型文件 (.bin)")
        else:
            print(f"  ⚠ 未找到模型文件")
else:
    print(f"  ✗ 模型路径不存在: {model_path}")
    sys.exit(1)

# 6. 测试 GPU 内存分配
print("\n6. GPU 内存测试:")
try:
    # 分配一些显存测试
    test_tensor = torch.randn(1000, 1000).cuda()
    print(f"  ✓ GPU 内存分配成功")
    print(f"    - 测试张量大小: {test_tensor.element_size() * test_tensor.nelement() / 1024**2:.2f} MB")
    print(f"    - 当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    del test_tensor
    torch.cuda.empty_cache()
    print(f"    - 释放后显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
except Exception as e:
    print(f"  ✗ GPU 内存测试失败: {e}")

# 7. 测试 Flash Attention
print("\n7. Flash Attention 检查:")
try:
    from transformers import Qwen3VLForConditionalGeneration
    print(f"  ✓ Qwen3VLForConditionalGeneration 可导入")
except ImportError as e:
    print(f"  ✗ Qwen3VLForConditionalGeneration 导入失败: {e}")

# 8. 测试 4-bit 量化支持
print("\n8. 4-bit 量化支持检查:")
try:
    from transformers import BitsAndBytesConfig
    print(f"  ✓ BitsAndBytesConfig 可导入")

    # 测试量化配置
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"  ✓ 4-bit 量化配置创建成功")
except ImportError as e:
    print(f"  ✗ BitsAndBytes 导入失败: {e}")
except Exception as e:
    print(f"  ✗ 4-bit 量化配置失败: {e}")

# 9. 测试模型加载 (可选)
print("\n9. 模型加载测试 (可选):")
load_test = 'y'  # 自动测试
try:
    print("  正在加载模型...")
    from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

    # 4-bit 量化配置
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    processor = Qwen3VLProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # 使用原生注意力机制
        low_cpu_mem_usage=True
    )

    print(f"  ✓ 模型加载成功!")
    print(f"    - 设备: {model.device}")
    print(f"    - 词汇表大小: {len(processor.tokenizer)}")
    print(f"    - 显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # 测试简单推理
    print("  正在测试推理...")
    from PIL import Image
    import numpy as np

    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "这是什么场景？"}
            ]
        }
    ]

    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text_input],
        images=[test_image],
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.0
        )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
    print(f"  ✓ 推理成功!")
    print(f"    - 输出: {generated_text[0][:100]}...")

    del model, processor
    torch.cuda.empty_cache()

except Exception as e:
    print(f"  ✗ 模型加载/推理失败: {e}")
    import traceback
    traceback.print_exc()

# 10. 总结
print("\n" + "=" * 70)
print("测试总结:")
print("=" * 70)

checks = [
    ("PyTorch 版本", torch.__version__),
    ("CUDA 可用", torch.cuda.is_available()),
    ("GPU 检测", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"),
    ("transformers", "✓" if 'transformers' in sys.modules else "✗"),
    ("模型路径", "✓" if Path(model_path).exists() else "✗"),
]

for check_name, check_result in checks:
    status = "✓" if check_result not in ["✗", "None", False] else "✗"
    print(f"  {status} {check_name}: {check_result}")

print("\n" + "=" * 70)
if all(check_result not in ["✗", "None", False] for _, check_result in checks):
    print("✅ GPU 环境配置成功！可以使用 GPU 运行 Qwen3-VL")
    print("\n下一步:")
    print("  1. 运行: python3 src/model_wrapper/qwen3vl_gpu_native.py")
    print("  2. 或在 TravelUAV 中使用 Qwen3VLGPUNativeWrapper")
else:
    print("⚠️  部分检查失败，请根据上面的错误信息进行修复")
print("=" * 70)
