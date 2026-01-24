"""
Qwen3-VL-4B-Instruct 集成 - GPU 优化版本

使用 PyTorch 2.9.1+cu128 + CUDA 12.8 + Blackwell GPU (RTX PRO 6000)
支持 4-bit 量化以减少显存占用
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.model_wrapper.base_model import BaseModelWrapper


class Qwen3VLGPUNativeWrapper(BaseModelWrapper):
    """
    GPU 优化版本 - 使用 transformers 4.57+ 直接加载 Qwen3-VL
    支持 4-bit 量化和 Flash Attention 2/3
    """

    def __init__(self, model_args, data_args, use_traj_model=True):
        print("正在加载 Qwen3-VL-4B-Instruct (GPU 优化版)...")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

        self.processor, self.model = self._load_model(model_args)
        self.use_traj_model = use_traj_model

        if use_traj_model:
            self.traj_model = self._load_traj_model(model_args)
            self.traj_model.to(dtype=torch.bfloat16, device=self.model.device)

        print(f"✓ 模型设备: {self.model.device}")
        print(f"✓ 显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        self.dino_monitor = None
        self.model_args = model_args
        self.data_args = data_args

        # 设置特殊 token
        self._setup_special_tokens()

        print("✓ Qwen3-VL-4B-Instruct (GPU) 加载完成！")

    def _load_model(self, model_args):
        """使用 transformers 直接加载 Qwen3-VL (GPU 优化)"""
        model_path = model_args.model_path
        use_4bit = getattr(model_args, 'use_4bit', True)

        print(f"模型路径: {model_path}")
        print(f"4-bit 量化: {use_4bit}")

        # 加载 Processor
        print("加载处理器...")
        from transformers import Qwen3VLProcessor
        processor = Qwen3VLProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        # 设置 padding_side='left' 以避免 decoder-only 架构的警告
        processor.tokenizer.padding_side = 'left'

        # 加载 Model - 使用 Qwen3VLForConditionalGeneration
        print("加载模型（可能需要几分钟）...")
        from transformers import Qwen3VLForConditionalGeneration

        # GPU 优化配置
        if use_4bit:
            print("使用 4-bit 量化 (QLoRA)")
            from transformers import BitsAndBytesConfig

            # 4-bit 量化配置
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
                dtype=torch.bfloat16,
                attn_implementation="eager",  # 使用原生注意力机制
                low_cpu_mem_usage=True
            )
            print("✓ 4-bit 量化模式加载成功")

        else:
            print("使用全精度 (FP16/BF16)")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",  # 使用原生注意力机制
                low_cpu_mem_usage=True
            )
            print("✓ 全精度模式加载成功")

        return processor, model

    def _load_traj_model(self, model_args):
        """加载轨迹预测模型 (GPU 优化)"""
        from src.model_wrapper.utils.travel_util_clean import (
            generate_vision_tower_config
        )
        import transformers

        # Add the llamavid-archive directory to sys.path for proper imports
        llamavid_dir = "/home/yyx/TravelUAV/Model/LLaMA-UAV/llamavid-archive"
        if llamavid_dir not in sys.path:
            sys.path.insert(0, llamavid_dir)

        # Add the model directory to sys.path as well
        model_dir = "/home/yyx/TravelUAV/Model/LLaMA-UAV/llamavid-archive/model"
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        # Import the module - this will trigger __init__.py but that's needed for relative imports
        from model.vis_traj_arch import VisionTrajectoryGenerator

        print("加载轨迹预测模型 (GPU)...")

        vision_config = generate_vision_tower_config(
            model_args.vision_tower,
            model_args.image_processor
        )
        config = transformers.AutoConfig.from_pretrained(
            vision_config,
            trust_remote_code=True
        )
        traj_model = VisionTrajectoryGenerator(config)

        # 加载权重
        traj_weight_path = f"{model_args.traj_model_path}/model_5.pth"
        print(f"加载轨迹权重: {traj_weight_path}")

        traj_weights = torch.load(traj_weight_path, map_location='cpu')
        traj_weights = {k: v.to(torch.bfloat16) for k, v in traj_weights.items()}
        traj_model.load_state_dict(traj_weights, strict=False)
        traj_model.to(device=self.model.device)

        return traj_model

    def _setup_special_tokens(self):
        """添加特殊 token"""
        special_tokens = ['<wp>', '<his>']

        num_new_tokens = self.processor.tokenizer.add_tokens(
            special_tokens,
            special_tokens=True
        )

        if num_new_tokens > 0:
            print(f"添加 {num_new_tokens} 个特殊 token")
            self.model.resize_token_embeddings(len(self.processor.tokenizer))

        self.special_token_dict = {
            '<wp>': self.processor.tokenizer.encode('<wp>')[0],
            '<his>': self.processor.tokenizer.encode('<his>')[0],
            ',': self.processor.tokenizer.encode(',')[0],
            ';': self.processor.tokenizer.encode(';')[0]
        }

        if hasattr(self.model, 'special_token_dict'):
            self.model.special_token_dict = self.special_token_dict

    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        """准备输入"""
        inputs = []
        rot_to_targets = []

        for i in range(len(episodes)):
            input_item, rot_to_target = self._prepare_single_input(
                episodes[i],
                target_positions[i],
                assist_notices[i] if assist_notices is not None else None
            )
            inputs.append(input_item)
            rot_to_targets.append(rot_to_target)

        batch = self._batch_inputs(inputs)
        return batch, rot_to_targets

    def _prepare_single_input(self, episode, target_point, assist_notice):
        """准备单个输入"""
        # 提取图像
        images = []
        for src in episode[::-1]:
            if 'rgb' in src:
                images.extend(src['rgb'])
                break

        # 确保图像数组是可写的，避免 NumPy 警告
        for i, img in enumerate(images):
            if hasattr(img, 'copy'):
                images[i] = img.copy()

        # 计算历史航点
        rot = np.array(episode[0]['sensors']['imu']["rotation"])
        pos = np.array(episode[0]['sensors']['state']['position'])

        deltas = []
        for src in episode:
            if 'rgb' not in src:
                continue
            deltas.append(np.array(src['sensors']['state']['position']) - pos)

        history_waypoint = np.array([(rot.T @ delta) for delta in deltas])

        # 计算到目标的旋转
        target_rel = np.array(rot.T @ (target_point - pos))
        rotation_to_target = self._rotation_matrix_from_vector(target_rel[0], target_rel[1])
        history_waypoint = np.dot(history_waypoint, rotation_to_target)

        # 构建提示
        stage = assist_notice if assist_notice else (
            'cruise' if len(episode) > 20 else 'take off'
        )

        if len(history_waypoint) >= 2:
            delta = history_waypoint[-1] - history_waypoint[-2]
        else:
            delta = np.array([0, 0, -4.5])

        delta = delta / (np.linalg.norm(delta) + 1e-8)
        delta_str = ','.join([str(round(x, 1)) for x in delta])
        cur_pos_str = ','.join([str(round(x, 1)) for x in history_waypoint[-1]])

        instruction = episode[-1]['instruction']

        # Qwen3-VL 提示格式
        prompt = (
            f"Stage: {stage}\n"
            f"Previous displacement: {delta_str}\n"
            f"Current position: {cur_pos_str}\n"
            f"Current image: <image>\n"
            f"Instruction: {instruction}\n"
            f"Output waypoints (x,y,z,norm):"
        )

        # 使用 Qwen 的 chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in images
                ] + [{"type": "text", "text": prompt}]
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 处理输入
        inputs = self.processor(
            text=[text_input],
            images=images,
            return_tensors="pt",
            padding=True
        )

        data_dict = {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'pixel_values': inputs['pixel_values'],
            'image_grid_thw': inputs['image_grid_thw'],
            'history_waypoint': torch.tensor(history_waypoint).view(-1),
            'orientation': torch.tensor(rotation_to_target).view(-1),
            'prompt': prompt,
            'images': images
        }

        return data_dict, rotation_to_target

    def _batch_inputs(self, instances):
        """批处理输入 - 添加padding处理"""
        # Find max length for padding
        max_len = max(inst['input_ids'].shape[-1] for inst in instances)

        # Pad input_ids and attention_mask to max length
        input_ids_list = []
        attention_mask_list = []
        for inst in instances:
            input_ids = inst['input_ids']
            attention_mask = inst['attention_mask']

            # Pad if necessary
            if input_ids.shape[-1] < max_len:
                pad_len = max_len - input_ids.shape[-1]
                # Handle both 1D and 2D tensors
                if input_ids.dim() == 1:
                    # 1D tensor: [seq_len]
                    pad_input_ids = torch.full((pad_len,),
                                               self.processor.tokenizer.pad_token_id,
                                               dtype=input_ids.dtype,
                                               device=input_ids.device)
                    pad_attention_mask = torch.zeros((pad_len,),
                                                     dtype=attention_mask.dtype,
                                                     device=attention_mask.device)
                else:
                    # 2D tensor: [batch, seq_len]
                    pad_input_ids = torch.full((input_ids.shape[0], pad_len),
                                               self.processor.tokenizer.pad_token_id,
                                               dtype=input_ids.dtype,
                                               device=input_ids.device)
                    pad_attention_mask = torch.zeros((attention_mask.shape[0], pad_len),
                                                     dtype=attention_mask.dtype,
                                                     device=attention_mask.device)
                input_ids = torch.cat([input_ids, pad_input_ids], dim=-1)
                attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=-1)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        # Pad history_waypoint to same size
        history_waypoints = [inst['history_waypoint'] for inst in instances]
        max_len = max(len(h) for h in history_waypoints)
        padded_history = []
        for h in history_waypoints:
            if len(h) < max_len:
                # Pad with zeros
                padded = torch.cat([h, torch.zeros(max_len - len(h), dtype=h.dtype, device=h.device)])
            else:
                padded = h
            padded_history.append(padded)

        batch = {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'pixel_values': torch.cat([inst['pixel_values'] for inst in instances]),
            'image_grid_thw': torch.cat([inst['image_grid_thw'] for inst in instances]),
            'prompts': [inst['prompt'] for inst in instances],
            'historys': torch.stack(padded_history),
            'orientations': torch.stack([inst['orientation'] for inst in instances]),
            'images': [inst['images'] for inst in instances],
            'return_waypoints': True,
            'use_cache': False
        }
        return batch

    def run_llm_model(self, inputs):
        """运行 Qwen 模型生成航点 (GPU 优化)"""
        with torch.no_grad():
            generate_kwargs = {
                'max_new_tokens': 100,
                'do_sample': False,
                'pad_token_id': self.processor.tokenizer.pad_token_id,
                'top_p': 1.0
            }

            # 生成
            outputs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.model.device),
                attention_mask=inputs['attention_mask'].to(self.model.device),
                pixel_values=inputs['pixel_values'].to(self.model.device),
                image_grid_thw=inputs['image_grid_thw'].to(self.model.device),
                **generate_kwargs
            )

            # 解码
            generated_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )

            # 解析航点
            waypoints_llm = []
            for i, text in enumerate(generated_text):
                prompt = inputs['prompts'][i]
                response = text[len(prompt):].strip()
                waypoint = self._parse_waypoint_output(response)
                waypoints_llm.append(waypoint)

            waypoints_llm = np.array(waypoints_llm)

            # 转换格式
            waypoints_llm_new = []
            for waypoint in waypoints_llm:
                norm = np.linalg.norm(waypoint[:3])
                if norm > 1e-6:
                    # Normalize the direction vector and scale by distance
                    direction = waypoint[:3] / norm
                    waypoint_new = np.array([direction[0], direction[1], direction[2], waypoint[3]])
                else:
                    waypoint_new = np.array([0, 0, 1, waypoint[3]])
                waypoints_llm_new.append(waypoint_new)

            return np.array(waypoints_llm_new)

    def _parse_waypoint_output(self, text):
        """解析文本输出为航点"""
        try:
            import re
            numbers = re.findall(r"[-+]?\d*\.?\d+|\\d+", text)
            if len(numbers) >= 4:
                return np.array([float(n) for n in numbers[:4]])
        except:
            pass

        return np.array([0, 0, 1, 1.0])

    def run_traj_model(self, episodes, waypoints_llm_new, rot_to_targets):
        """运行轨迹预测模型 (GPU 优化)"""
        if not self.use_traj_model:
            waypoints_world = []
            for i, waypoint in enumerate(waypoints_llm_new):
                ep = episodes[i]
                pos = ep[-1]["sensors"]["state"]["position"]
                rot = ep[-1]["sensors"]["imu"]["rotation"]
                waypoint_world = np.array(rot) @ np.array(waypoint[:3]) + np.asarray(pos)
                waypoints_world.append(waypoint_world)
            return np.array(waypoints_world)

        from src.model_wrapper.utils.travel_util_clean import (
            prepare_data_to_traj_model,
            transform_to_world
        )

        # Use local CLIP image processor for the trajectory model (EVA-ViT expects CLIP processor)
        from transformers import CLIPImageProcessor
        clip_processor_path = "/home/yyx/TravelUAV/Model/LLaMA-UAV/llamavid-archive/processor/clip-patch14-224"
        image_processor = CLIPImageProcessor.from_pretrained(clip_processor_path)

        inputs = prepare_data_to_traj_model(
            episodes,
            waypoints_llm_new,
            image_processor,
            rot_to_targets
        )
        waypoints_traj = self.traj_model(inputs, None)
        # 修复: 使用 detach() 移除梯度，避免 RuntimeError
        refined_waypoints = waypoints_traj.detach().cpu().to(dtype=torch.float32).numpy()
        refined_waypoints = transform_to_world(refined_waypoints, episodes)
        return refined_waypoints

    def eval(self):
        """设置评估模式"""
        self.model.eval()
        if self.use_traj_model:
            self.traj_model.eval()

    def run(self, inputs, episodes, rot_to_targets):
        """完整运行"""
        waypoints_llm_new = self.run_llm_model(inputs)
        refined_waypoints = self.run_traj_model(
            episodes,
            waypoints_llm_new,
            rot_to_targets
        )
        return refined_waypoints

    def predict_done(self, episodes, object_infos):
        """预测完成状态"""
        prediction_dones = []

        if self.dino_monitor is None:
            try:
                from src.vlnce_src.dino_monitor_online import DinoMonitor
                self.dino_monitor = DinoMonitor.get_instance()
            except Exception as e:
                print(f"⚠ DinoMonitor 初始化失败: {e}")
                print("  跳过完成状态预测")
                return [False] * len(episodes)

        for i in range(len(episodes)):
            prediction_done = self.dino_monitor.get_dino_results(
                episodes[i],
                object_infos[i]
            )
            prediction_dones.append(prediction_done)

        return prediction_dones

    @staticmethod
    def _rotation_matrix_from_vector(x, y):
        """从向量创建旋转矩阵"""
        v_x = np.array([x, y, 0])
        v_x = v_x / np.linalg.norm(v_x)
        v_y = np.array([-v_x[1], v_x[0], 0])
        v_y = v_y / np.linalg.norm(v_y)
        v_z = np.array([0, 0, 1])
        return np.column_stack((v_x, v_y, v_z))


def test_qwen3vl_gpu():
    """测试 GPU 版本"""
    print("=" * 60)
    print("测试 Qwen3-VL GPU 模式")
    print("=" * 60)

    model_path = "/home/yyx/TravelUAV/Model/Qwen3-VL-4B-Instruct"

    if not Path(model_path).exists():
        print(f"✗ 模型目录不存在")
        return False

    print(f"✓ 模型路径: {model_path}")

    try:
        from dataclasses import dataclass

        @dataclass
        class ModelArgs:
            model_path: str = "/home/yyx/TravelUAV/Model/Qwen3-VL-4B-Instruct"
            model_base: str = None
            traj_model_path: str = "/home/yyx/TravelUAV/Model/LLaMA-UAV/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4"
            vision_tower: str = "/home/yyx/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth"
            image_processor: str = "/home/yyx/TravelUAV/Model/LLaMA-UAV/llamavid-archive/processor/clip-patch14-224"
            use_4bit: bool = True

        @dataclass
        class DataArgs:
            input_prompt: str = None
            refine_prompt: bool = True

        print("\n初始化 GPU 包装器...")
        model_args = ModelArgs()
        data_args = DataArgs()

        # 测试加载
        wrapper = Qwen3VLGPUNativeWrapper(model_args, data_args, use_traj_model=False)

        # 测试简单推理
        print("\n测试简单推理...")
        from PIL import Image
        import numpy as np

        # 创建测试图像
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        # 创建测试输入
        test_episode = [{
            'instruction': '向前飞行',
            'rgb': [test_image],
            'sensors': {
                'imu': {'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                'state': {'position': [0, 0, 0]}
            }
        }]
        test_target = [10, 0, 5]

        inputs, rot_to_targets = wrapper.prepare_inputs([test_episode], [test_target])
        waypoints = wrapper.run_llm_model(inputs)

        print(f"  ✓ 输入准备成功")
        print(f"  ✓ LLM 推理成功")
        print(f"    - 输出航点: {waypoints}")

        print(f"\n✓ 成功！")
        print(f"  设备: {wrapper.model.device}")
        print(f"  词汇表: {len(wrapper.processor.tokenizer)}")
        print(f"  显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        return True

    except Exception as e:
        print(f"\n✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_qwen3vl_gpu()
