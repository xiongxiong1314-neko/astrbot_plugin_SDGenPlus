import asyncio
import json
import math
import os
import re
import shutil
import time

import aiohttp

from astrbot.api.all import *

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    import numpy as _np
except ImportError:
    _np = None


TEMP_PATH = os.path.abspath("data/temp")
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

# 词库向量索引缓存（词库文件的向量化结果，按模型/api/mtime 失效自动重建）
EMBEDDING_CACHE_DIR = os.path.join(os.path.abspath("data"), "sdgen")
EMBEDDING_CACHE_FILE = os.path.join(EMBEDDING_CACHE_DIR, "vocab_embedding_index.json")
EMBEDDING_BATCH_SIZE = 32

# OpenAI 兼容提供商适配器注册名 -> 推荐的默认 embedding 模型（type 见 AstrBot 提供商注册名）
EMBEDDING_DEFAULT_MODELS = {
    "openai_chat_completion": "text-embedding-3-small",
    "aihubmix_chat_completion": "text-embedding-3-small",
}

@register("SDGenPlus", "xiongxiong", "Stable Diffusion图像生成器(集成标准词库+新模型支持)", "1.0.1")
class SDGenerator(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.session = None
        self._session_lock = asyncio.Lock()
        self._validate_config()
        os.makedirs(TEMP_PATH, exist_ok=True)

        # 初始化并发控制
        self.active_tasks = 0
        self.max_concurrent_tasks = config.get("max_concurrent_tasks", 10)  # 设定最大并发数
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # 标准词库索引缓存
        self._vocab_index = None
        self._vocab_index_mtime = None

        # 词库向量检索（embedding）状态
        self._embed_state = "idle"      # idle | building | ready | error
        self._embed_data = None         # {entries, vectors, model, api_base, mtime, dim}
        self._embed_error = None
        self._embed_lock = asyncio.Lock()
        self._embed_last_try = 0.0
        self._embed_build_task = None    # 后台构建任务引用（用于取消）
        self._embed_client = None          # 复用的 AsyncOpenAI 客户端
        self._embed_client_key = None      # (api_base, api_key) 用于判断复用

        # 首次加载时把随插件自带的词库分发到 data 目录（不覆盖已存在文件）
        self._distribute_bundled_vocab()

        # 启动时后台构建词库向量索引（加载缓存很快；首次构建需一段时间，不阻塞启动）
        self._spawn_embedding_index_build()

    def _spawn_embedding_index_build(self):
        """尽力在事件循环上安排后台构建任务；事件循环未运行时推迟到首次检索时构建。"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._embed_build_task = loop.create_task(
                    self._ensure_embedding_index(force=False)
                )
        except Exception as e:
            logger.debug(f"暂无法在启动时构建词库向量索引: {e}")

    def _cancel_embedding_build(self):
        """取消正在进行的后台构建任务，并返回原任务供调用方等待收尾。"""
        task = self._embed_build_task
        self._embed_build_task = None
        if task is not None and not task.done():
            task.cancel()
        return task

    def _distribute_bundled_vocab(self):
        """若目标词库文件不存在，则把插件自带的 prompt_vocabulary.txt 复制过去"""
        target = (self.config.get("prompt_vocabulary_path") or "").strip()
        if not target:
            return
        if not os.path.isabs(target):
            target = os.path.abspath(target)
        if os.path.exists(target):
            return
        bundled = os.path.join(PLUGIN_DIR, "prompt_vocabulary.txt")
        if not os.path.exists(bundled):
            return
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(bundled, target)
            logger.info(f"已将随插件自带的词库分发到: {target}")
        except Exception as e:
            logger.error(f"分发自带词库失败: {e}")

    @staticmethod
    def _select_prompt_option(group: dict, index_key: str, prefix: str, count: int = 4) -> str:
        """Select prompt by index with safe fallback."""
        index = group.get(index_key, 0)
        if not isinstance(index, int) or index < 0 or index >= count:
            index = 0
        return group.get(f"{prefix}{index}", "")

    @staticmethod
    def _compose_prompt(*segments: str) -> str:
        """Join non-empty prompt segments with commas."""
        return ",".join(segment for segment in segments if segment)

    def _validate_config(self):
        """配置验证"""
        self.config["webui_url"] = self.config["webui_url"].strip()
        if not self.config["webui_url"].startswith(("http://", "https://")):
            raise ValueError("WebUI地址必须以http://或https://开头")

        if self.config["webui_url"].endswith("/"):
            self.config["webui_url"] = self.config["webui_url"].rstrip("/")
            self.config.save_config()

    async def ensure_session(self):
        """确保共享 HTTP 会话可用，避免并发请求重复创建连接池。"""
        if self.session is not None and not self.session.closed:
            return self.session
        async with self._session_lock:
            if self.session is None or self.session.closed:
                timeout = max(10, int(self.config.get("session_timeout_time", 120)))
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout)
                )
        return self.session

    async def terminate(self):
        """插件卸载时释放后台任务、HTTP 会话与兼容 embedding 客户端。"""
        task = self._cancel_embedding_build()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"等待词库索引任务结束时出现异常: {e}")

        if self.session is not None and not self.session.closed:
            await self.session.close()
        self.session = None

        if self._embed_client is not None:
            try:
                await self._embed_client.close()
            except Exception as e:
                logger.debug(f"关闭 embedding 客户端失败: {e}")
        self._embed_client = None
        self._embed_client_key = None

    async def _fetch_webui_resource(self, resource_type: str) -> list:
        """从 WebUI API 获取指定类型的资源列表"""
        endpoint_map = {
            "model": "/sdapi/v1/sd-models",
            "embedding": "/sdapi/v1/embeddings",
            "lora": "/sdapi/v1/loras",
            "sampler": "/sdapi/v1/samplers",
            "upscaler": "/sdapi/v1/upscalers"
        }
        if resource_type not in endpoint_map:
            logger.error(f"无效的资源类型: {resource_type}")
            return []

        try:
            await self.ensure_session()
            async with self.session.get(f"{self.config['webui_url']}{endpoint_map[resource_type]}") as resp:
                if resp.status == 200:
                    resources = await resp.json()

                    # 按不同类型解析返回数据
                    if resource_type == "model":
                        resource_names = [r["model_name"] for r in resources if "model_name" in r]
                    elif resource_type == "embedding":
                        resource_names = list(resources.get('loaded', {}).keys())
                    elif resource_type == "lora":
                        resource_names = [r["name"] for r in resources if "name" in r]
                    elif resource_type == "sampler":
                        resource_names = [r["name"] for r in resources if "name" in r]
                    elif resource_type == "upscaler":
                        resource_names = [r["name"] for r in resources if "name" in r]

                    else:
                        resource_names = []

                    logger.debug(f"从 WebUI 获取到的{resource_type}资源: {resource_names}")
                    return resource_names
        except Exception as e:
            logger.error(f"获取 {resource_type} 类型资源失败: {e}")

        return []

    async def _get_sd_model_list(self):
        return await self._fetch_webui_resource("model")

    async def _get_embedding_list(self):
        return await self._fetch_webui_resource("embedding")

    async def _get_lora_list(self):
        return await self._fetch_webui_resource("lora")

    async def _get_sampler_list(self):
        """获取可用的采样器列表"""
        return await self._fetch_webui_resource("sampler")

    async def _get_upscaler_list(self):
        """获取可用的上采样算法列表"""
        return await self._fetch_webui_resource("upscaler")

    async def _get_webui_options(self) -> dict:
        """获取 WebUI 当前配置（当前模型、VAE 等）"""
        try:
            await self.ensure_session()
            async with self.session.get(f"{self.config['webui_url']}/sdapi/v1/options") as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error(f"获取 WebUI options 失败 (状态码: {resp.status})")
        except Exception as e:
            logger.error(f"获取 WebUI options 异常: {e}")
        return {}

    async def _get_current_webui_model_info(self) -> tuple[str, str]:
        """自动识别 WebUI 当前加载的基础模型和 VAE"""
        options = await self._get_webui_options()
        model = options.get("sd_model_checkpoint") or options.get("sd_checkpoint_hash") or "未识别"
        vae = options.get("sd_vae") or "未识别"
        return model, vae

    def _build_negative_prompt(self) -> str:
        """Assemble negative prompt from global preset."""
        global_group = self.config.get("global_prompt_group", {})

        return (
            global_group.get("global_negative_prompt", "")
            if global_group.get("global_negative_prompt_switch", False)
            else ""
        )

    def _build_lora_tags(self) -> list:
        """根据配置解析默认 LoRA 列表，返回 SD WebUI 的 <lora:name:weight> 语法串列表。

        配置格式（new_model_params.lora，逗号分隔多个）：
            lora名:权重 或 lora名  （缺省权重为 1.0）
        示例：chibi:0.8, detail_slider:0.5
        """
        new_params = self.config.get("new_model_params", {})
        raw = (new_params.get("lora") or "").strip()
        if not raw:
            return []
        tags = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            name = parts[0].strip()
            if not name:
                continue
            weight = 1.0
            if len(parts) > 1 and parts[1].strip():
                try:
                    weight = float(parts[1].strip())
                    if weight <= 0:
                        weight = 1.0
                except ValueError:
                    weight = 1.0
            tags.append(f"<lora:{name}:{weight}>")
        return tags

    def _compose_lora_prompt(self, prompt: str) -> str:
        """把默认 LoRA 拼到正面提示词最前面；若 prompt 已含同名 LoRA 则跳过，避免重复叠加。"""
        lora_tags = self._build_lora_tags()
        if not lora_tags:
            return prompt
        existing = {m.lower() for m in re.findall(r"<lora:([^:>]+)", prompt)}
        keep = [
            tag for tag in lora_tags
            if re.match(r"<lora:([^:>]+)", tag).group(1).lower() not in existing
        ]
        if not keep:
            return prompt
        return ", ".join(keep) + ", " + prompt

    async def _generate_payload(self, prompt: str) -> dict:
        """构建生成参数"""
        params = self.config["default_params"]
        negative_prompt = self._build_negative_prompt()
        new_params = self.config.get("new_model_params", {})
        prompt = self._compose_lora_prompt(prompt)

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": params["width"],
            "height": params["height"],
            "steps": params["steps"],
            "sampler_name": params["sampler"],
            "cfg_scale": params["cfg_scale"],
            "batch_size": params["batch_size"],
            "n_iter": params["n_iter"],
            "clip_skip": new_params.get("clip_skip", 0),
        }

        # SDXL Refiner 支持
        refiner = (new_params.get("refiner_checkpoint") or "").strip()
        if refiner:
            payload["refiner_checkpoint"] = refiner
            payload["refiner_switch_at"] = new_params.get("refiner_switch_at", 0.8)

        return payload

    def _trans_prompt(self, prompt: str) -> str:
        """返回原始提示词（保留空格）"""
        return prompt

    @staticmethod
    def _extract_prompt_from_message(event: AstrMessageEvent, raw_prompt: str) -> str:
        """从原始消息还原提示词，避免参数解析截断空格"""
        full = (event.message_str or "").strip()
        base = (raw_prompt or "").strip()

        if not full:
            return base

        tokens = full.split()
        if tokens and tokens[0].lstrip("/") in ("sd",):
            tokens = tokens[1:]
        if tokens and tokens[0] == "gen":
            tokens = tokens[1:]

        fallback = " ".join(tokens).strip()
        return fallback or base

    def _build_positive_prompt(self, raw_prompt: str, generated_prompt: str) -> str:
        """Construct final positive prompt with global preset."""
        global_group = self.config.get("global_prompt_group", {})

        global_positive_prompt = (
            global_group.get("global_positive_prompt", "")
            if global_group.get("global_positive_prompt_switch", False)
            else ""
        )
        add_global_first = global_group.get("positive_prompt_add_in_head_or_tail_switch", False)

        base_prompt = (
            generated_prompt if self.config.get("enable_generate_prompt") and generated_prompt else self._trans_prompt(raw_prompt)
        )

        if add_global_first:
            return self._compose_prompt(global_positive_prompt, base_prompt)
        return self._compose_prompt(base_prompt, global_positive_prompt)

    # ---- 标准词库：文件读取 / 解析 / 关键词检索 ----
    def _vocab_raw_text(self) -> str:
        """读取词库文件原始文本（不缓存，仅用于判断存在性与解析）"""
        path = (self.config.get("prompt_vocabulary_path") or "").strip()
        if not path:
            return ""
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        try:
            if not os.path.exists(path):
                return ""
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取标准词库文件失败: {e}")
            return ""

    def _vocab_mtime(self) -> float | None:
        """返回当前词库文件 mtime，用于内存索引与向量缓存失效判断。"""
        path = (self.config.get("prompt_vocabulary_path") or "").strip()
        if not path:
            return None
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    def _get_vocab_index(self) -> list:
        """获取解析后的词库条目列表，带内存缓存（按文件 mtime 自动失效）。返回 [(title, content), ...]"""
        path = (self.config.get("prompt_vocabulary_path") or "").strip()
        if not path:
            self._vocab_index = []
            self._vocab_index_mtime = None
            return self._vocab_index
        mtime = self._vocab_mtime()
        if self._vocab_index is not None and self._vocab_index_mtime == mtime:
            return self._vocab_index
        text = self._vocab_raw_text()
        if not text:
            self._vocab_index = []
            self._vocab_index_mtime = mtime
            return self._vocab_index
        self._vocab_index = self._parse_vocab_sections(text)
        self._vocab_index_mtime = mtime
        logger.debug(f"标准词库已解析: {len(self._vocab_index)} 条目")
        return self._vocab_index

    @staticmethod
    def _cn_ratio(s: str) -> float:
        s = s.strip()
        if not s:
            return 0.0
        cn = sum(1 for c in s if '\u4e00' <= c <= '\u9fff')
        return cn / len(s)

    @staticmethod
    def _is_vocab_title(line: str) -> bool:
        """判断一行是否为词库的分类/场景标题（中文短行）"""
        s = line.strip()
        if not s or len(s) > 40:
            return False
        if ',' in s or '，' in s:
            return False
        if s.startswith(('ps', 'PS', 'Ps', '—', 'NAI', 'nai', '原版', 'char', 'Char', ':', '：')):
            return False
        if re.match(r'^\d', s):
            return False
        return SDGenerator._cn_ratio(s) >= 0.5

    def _parse_vocab_sections(self, text: str) -> list:
        """把词库文本切成 [(title, content), ...]，自动跳过前言与目录"""
        lines = text.split('\n')
        start = 0
        for i, ln in enumerate(lines):
            if ln.strip() == "目录":
                start = i + 1
                break
        # 跳过目录区（以数字结尾的连续行）
        while start < len(lines) and re.search(r'\d+$', lines[start].strip()):
            start += 1
        entries = []
        cur_title = None
        cur_content = []
        for ln in lines[start:]:
            s = ln.strip()
            if not s:
                continue
            if self._is_vocab_title(ln):
                if cur_title is not None or cur_content:
                    entries.append((cur_title or "", "\n".join(cur_content)))
                cur_title = s
                cur_content = []
            else:
                cur_content.append(s)
        if cur_title is not None or cur_content:
            entries.append((cur_title or "", "\n".join(cur_content)))
        return entries

    @staticmethod
    def _normalize_api_base(api_base: str) -> str:
        """规范化 OpenAI 兼容 api_base：去尾部斜杠与 /embeddings，无 /v 后缀时补 /v1。"""
        api_base = (api_base or "").strip().removesuffix("/").removesuffix("/embeddings")
        if api_base and not re.search(r"/v\d+$", api_base):
            api_base = api_base + "/v1"
        return api_base

    def _get_embedding_provider(self):
        """定位词库向量化的 embedding 提供方。

        优先级：
        1. 配置指定了 embedding_provider_id -> 按 id 从 inst_map 取（可能是 EmbeddingProvider 或 chat Provider）；
        2. 未指定 -> 自动使用 AstrBot 已配置的第一个 EmbeddingProvider（模型来自 AstrBot 提供商配置）；
        3. 都没有 -> 回退到当前对话提供商（仅 OpenAI 兼容路径，需手动指定 embedding_model）。

        Returns:
            (provider, api_base, api_key)：原生 EmbeddingProvider 时 api_base/api_key 为空串。
        """
        provider = None
        pid = (self.config.get("embedding_provider_id") or "").strip()
        if pid:
            get_by_id = getattr(self.context, "get_provider_by_id", None) or getattr(
                self.context, "get_provider", None
            )
            if get_by_id is not None:
                try:
                    provider = get_by_id(pid)
                except Exception:
                    provider = None
        if provider is None:
            get_all_eps = getattr(self.context, "get_all_embedding_providers", None)
            if get_all_eps is not None:
                try:
                    all_eps = get_all_eps()
                except Exception:
                    all_eps = []
                if all_eps:
                    provider = all_eps[0]
        if provider is None:
            try:
                provider = self.context.get_using_provider()
            except Exception:
                provider = None
        if provider is None:
            return None, "", ""
        # 原生 EmbeddingProvider：由 AstrBot 自己管理 api/模型，插件直接调接口
        if hasattr(provider, "get_embeddings"):
            return provider, "", ""
        # OpenAI 兼容回退路径：复用 chat provider 的 api_base/key
        pcfg = getattr(provider, "provider_config", None) or {}
        api_base = self._normalize_api_base(pcfg.get("api_base") or pcfg.get("base_url") or "")
        if not api_base:
            # OpenAI 官方提供商不填 api_base 时使用官方默认地址
            api_base = "https://api.openai.com/v1"
        keys = pcfg.get("key") or []
        if isinstance(keys, list):
            api_key = keys[0] if keys else ""
        else:
            api_key = str(keys or "")
        return provider, api_base, api_key

    def _resolve_embedding_model(self, provider) -> str:
        """确定 embedding 模型名。

        - 原生 EmbeddingProvider：直接使用 AstrBot 提供商配置的 embedding_model；
        - OpenAI 兼容回退：优先用户配置；其次提供商配置中含 embed 的字段；最后按提供商类型给默认值。
        """
        if hasattr(provider, "get_embeddings"):
            pcfg = getattr(provider, "provider_config", None) or {}
            return (
                str(pcfg.get("embedding_model") or "").strip()
                or str(getattr(provider, "model_name", "") or "").strip()
                or "embedding"
            )
        model = (self.config.get("embedding_model") or "").strip()
        if model and model.lower() != "auto":
            return model
        pcfg = getattr(provider, "provider_config", None) or {}
        for k, v in pcfg.items():
            if "embed" in str(k).lower() and isinstance(v, str) and v.strip():
                return v.strip()
        ptype = str(pcfg.get("type") or "")
        return self.EMBEDDING_DEFAULT_MODELS.get(ptype, "")

    def _embedding_provider_key(self, provider, api_base: str, model: str) -> str:
        """生成缓存失效用的提供商标识：原生 embedding provider 用 type|id|model，兼容路径再加 api_base。"""
        pcfg = getattr(provider, "provider_config", None) or {}
        ptype = str(pcfg.get("type") or "?")
        pid = str(pcfg.get("id") or "?")
        if hasattr(provider, "get_embeddings"):
            return f"native|{ptype}|{pid}|{model}"
        return f"compat|{ptype}|{pid}|{model}|{api_base}"

    async def _embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        """批量向量化文本。优先走 AstrBot 原生 EmbeddingProvider 接口；否则走 OpenAI 兼容回退。

        失败返回 None 并记录 self._embed_error。
        """
        provider, api_base, api_key = self._get_embedding_provider()
        if provider is None:
            self._embed_error = "未找到可用的 embedding 提供商，请在 AstrBot 提供商管理中配置 Embedding 提供商"
            return None
        # 原生 EmbeddingProvider：模型/密钥由 AstrBot 统一管理
        if hasattr(provider, "get_embeddings"):
            batch_method = getattr(provider, "get_embeddings_batch", None)
            try:
                if batch_method is not None:
                    return await batch_method(
                        texts, batch_size=self.EMBEDDING_BATCH_SIZE
                    )
                return await provider.get_embeddings(texts)
            except Exception as e:
                self._embed_error = f"embedding 请求失败: {e}"
                logger.error(f"embedding 请求失败: {e}")
                return None
        # OpenAI 兼容回退路径
        if not api_base:
            self._embed_error = "未找到可用的模型提供商（api_base 为空）"
            return None
        model = self._resolve_embedding_model(provider)
        if not model:
            ptype = str((getattr(provider, "provider_config", None) or {}).get("type") or "未知")
            self._embed_error = f"无法自动确定 embedding 模型（提供商类型: {ptype}），请在插件配置中手动指定"
            return None
        if AsyncOpenAI is None:
            self._embed_error = "缺少 openai 库，无法调用 embeddings 接口"
            return None
        client_key = (api_base, api_key)
        if self._embed_client is None or self._embed_client_key != client_key:
            if self._embed_client is not None:
                try:
                    await self._embed_client.close()
                except Exception:
                    pass
            self._embed_client = AsyncOpenAI(
                api_key=api_key or "EMPTY", base_url=api_base, timeout=120
            )
            self._embed_client_key = client_key
        client = self._embed_client
        try:
            resp = await client.embeddings.create(input=texts, model=model)
            return [d.embedding for d in resp.data]
        except Exception as e:
            self._embed_error = f"embedding 请求失败: {e}"
            logger.error(f"embedding 请求失败: {e}")
            return None

    async def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """分批向量化，避免单次请求过大。"""
        results: list[list[float]] = []
        total = len(texts)
        for i in range(0, total, self.EMBEDDING_BATCH_SIZE):
            chunk = texts[i:i + self.EMBEDDING_BATCH_SIZE]
            vecs = await self._embed_texts(chunk)
            if vecs is None:
                raise RuntimeError(self._embed_error or "embedding 请求失败")
            results.extend(vecs)
        return results

    @staticmethod
    def _normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
        """L2 归一化，检索时直接点积即余弦相似度。"""
        normed = []
        for vec in vectors:
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                normed.append([x / norm for x in vec])
            else:
                normed.append(vec)
        return normed

    def _load_embedding_cache(self) -> dict | None:
        try:
            if not os.path.exists(self.EMBEDDING_CACHE_FILE):
                return None
            with open(self.EMBEDDING_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"读取词库向量缓存失败: {e}")
            return None

    def _save_embedding_cache(self, data: dict) -> None:
        try:
            os.makedirs(self.EMBEDDING_CACHE_DIR, exist_ok=True)
            tmp = self.EMBEDDING_CACHE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.EMBEDDING_CACHE_FILE)
        except Exception as e:
            logger.warning(f"保存词库向量缓存失败: {e}")

    async def _ensure_embedding_index(self, force: bool = False) -> None:
        """确保词库向量索引就绪（幂等，带锁）。优先从本地缓存加载，缓存失效则重建。"""
        if self._embed_state == "building":
            return
        if not force and self._embed_state == "ready":
            return
        if not force and self._embed_state == "error":
            # 失败冷却 60 秒，避免每次生图都重复探测失败
            if time.time() - self._embed_last_try < 60:
                return
        async with self._embed_lock:
            if self._embed_state == "building":
                return
            if not force and self._embed_state == "ready":
                return
            entries = self._get_vocab_index()
            if not entries:
                self._embed_state = "idle"
                self._embed_data = None
                return
            provider, api_base, _ = self._get_embedding_provider()
            if provider is None:
                self._embed_state = "error"
                self._embed_error = "未找到可用的 embedding 提供商，请在 AstrBot 提供商管理中配置 Embedding 提供商"
                self._embed_last_try = time.time()
                return
            model = self._resolve_embedding_model(provider)
            if not model:
                self._embed_state = "error"
                self._embed_error = "无法自动确定 embedding 模型，请在插件配置中手动指定"
                self._embed_last_try = time.time()
                return
            provider_key = self._embedding_provider_key(provider, api_base, model)
            mtime = self._vocab_mtime()
            # 命中缓存则直接加载（校验提供商/模型/词库 mtime/条目数）
            if not force:
                cache = self._load_embedding_cache()
                if cache and cache.get("meta", {}).get("provider_key") == provider_key \
                        and cache.get("meta", {}).get("mtime") == mtime \
                        and len(cache.get("entries", [])) == len(entries):
                    vectors = cache.get("vectors") or []
                    if vectors:
                        self._embed_data = {
                            "entries": cache["entries"],
                            "vectors": self._normalize_vectors(vectors),
                            "provider_key": provider_key,
                            "model": model,
                            "api_base": api_base,
                            "mtime": mtime,
                            "dim": cache.get("meta", {}).get("dim", len(vectors[0])),
                        }
                        self._embed_state = "ready"
                        self._embed_last_try = time.time()
                        logger.info(f"词库向量索引已从缓存加载: {len(entries)} 条, 模型 {model}")
                        return
            # 重建
            self._embed_state = "building"
            self._embed_error = None
            started = time.time()
            try:
                texts = [f"{title}\n{content}" for title, content in entries]
                vectors = await self._embed_in_batches(texts)
                if not vectors:
                    raise RuntimeError("embedding 返回为空")
                dim = len(vectors[0])
                norm_vectors = self._normalize_vectors(vectors)
                self._embed_data = {
                    "entries": entries,
                    "vectors": norm_vectors,
                    "provider_key": provider_key,
                    "model": model,
                    "api_base": api_base,
                    "mtime": mtime,
                    "dim": dim,
                }
                self._save_embedding_cache({
                    "meta": {
                        "provider_key": provider_key,
                        "model": model,
                        "api_base": api_base,
                        "mtime": mtime,
                        "dim": dim,
                        "built_at": time.time(),
                    },
                    "entries": entries,
                    "vectors": norm_vectors,
                })
                logger.info(f"词库向量索引构建完成: {len(entries)} 条, 维度 {dim}, 耗时 {time.time() - started:.1f}s")
                self._embed_state = "ready"
                self._embed_last_try = time.time()
            except asyncio.CancelledError:
                self._embed_data = None
                self._embed_state = "idle"
                self._embed_error = None
                logger.info("词库向量索引构建已取消")
                raise
            except Exception as e:
                self._embed_data = None
                self._embed_state = "error"
                self._embed_error = str(e)
                self._embed_last_try = time.time()
                logger.error(f"词库向量索引构建失败: {e}")

    async def _embed_one(self, text: str) -> list[float] | None:
        """单条文本向量化（检索查询用）。"""
        provider, _, _ = self._get_embedding_provider()
        if provider is None:
            return None
        if hasattr(provider, "get_embedding"):
            try:
                return await provider.get_embedding(text)
            except Exception as e:
                self._embed_error = f"embedding 请求失败: {e}"
                logger.error(f"embedding 请求失败: {e}")
                return None
        res = await self._embed_texts([text])
        return res[0] if res else None

    def _top_similar(self, query_vec: list[float], k: int) -> list[tuple[float, int]]:
        """返回与查询向量最相似的 k 个 (相似度, 条目下标)，向量已归一化，点积即余弦。"""
        norm = math.sqrt(sum(x * x for x in query_vec)) or 1.0
        q = [x / norm for x in query_vec]
        vectors = self._embed_data["vectors"]
        if _np is not None:
            try:
                qv = _np.asarray(q, dtype=_np.float32)
                mat = _np.asarray(vectors, dtype=_np.float32)
                scores = mat @ qv
                order = _np.argsort(-scores)[:k].tolist()
                return [(float(scores[i]), i) for i in order]
            except Exception:
                pass
        scored = []
        for i, vec in enumerate(vectors):
            s = 0.0
            for a, b in zip(q, vec):
                s += a * b
            scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    async def _retrieve_vocab(self, query: str) -> str:
        """根据用户描述用 embedding 余弦相似度检索词库片段，返回拼好的字符串（无命中返回空）。"""
        if not self.config.get("embedding_enabled", True):
            return ""
        provider, api_base, _ = self._get_embedding_provider()
        if provider is None:
            self._embed_state = "error"
            self._embed_data = None
            self._embed_error = "未找到可用的 embedding 提供商"
            return ""
        model = self._resolve_embedding_model(provider)
        current_provider_key = self._embedding_provider_key(provider, api_base, model)
        index_stale = (
            self._embed_state == "ready"
            and self._embed_data is not None
            and (
                self._embed_data.get("provider_key") != current_provider_key
                or self._embed_data.get("mtime") != self._vocab_mtime()
                or len(self._embed_data.get("entries", [])) != len(self._get_vocab_index())
            )
        )
        if index_stale:
            self._embed_state = "idle"
            self._embed_data = None
        if self._embed_state != "ready":
            # 未就绪时确保有后台任务在构建，本次不阻塞生图、不注入词库
            if self._embed_build_task is None or self._embed_build_task.done():
                try:
                    self._embed_build_task = asyncio.get_running_loop().create_task(
                        self._ensure_embedding_index(force=False)
                    )
                except Exception:
                    pass
            if self._embed_state != "ready":
                if self._embed_error:
                    logger.warning(f"词库向量检索不可用: {self._embed_error}")
                return ""
        qv = await self._embed_one(query)
        if not qv:
            return ""
        top_k = max(1, int(self.config.get("prompt_vocabulary_top_k", 8)))
        max_chars = max(1, int(self.config.get("prompt_vocabulary_max_chars", 4000)))
        hits = self._top_similar(qv, top_k)
        entries = self._embed_data["entries"]
        snippets = []
        total = 0
        for score, idx in hits:
            if score <= 0.0:
                continue
            title, content = entries[idx]
            snippet = f"【{title}】\n{content}" if content else f"【{title}】"
            if total + len(snippet) > max_chars:
                break
            snippets.append(snippet)
            total += len(snippet)
        return "\n\n".join(snippets)

    async def _generate_prompt(self, prompt: str) -> str:
        provider = self.context.get_using_provider()
        if provider:
            prompt_guidelines = self.config.get("prompt_guidelines", "")
            prompt_vocabulary = await self._retrieve_vocab(prompt)
            prompt_generate_text = (
                "请根据以下描述生成用于 Stable Diffusion WebUI 的英文提示词，"
                "请返回一条逗号分隔的 `prompt` 英文字符串，适用于 Stable Diffusion web UI，"
                "其中应包含主体、风格、光照、色彩等方面的描述，"
                "避免解释性文本，不需要 “prompt:” 等内容，不需要双引号包裹，"
                "直接返回 `prompt`，不要加任何额外说明。"
            )
            if prompt_vocabulary:
                prompt_generate_text += (
                    "\n请优先参考以下从标准词库中检索到的相关词条与示例组合，"
                    "尽量复用其中贴合描述的英文 tag、权重写法（如 {} [] :: 等）与风格，"
                    "可在此基础上增补但不要抛弃词库中的规范词条：\n"
                    f"{prompt_vocabulary}\n"
                )
            if prompt_guidelines:
                prompt_generate_text += f"\n{prompt_guidelines}\n"
            prompt_generate_text += "描述："

            response = await provider.text_chat(f"{prompt_generate_text} {prompt}", session_id=None)
            if response.completion_text:
                generated_prompt = re.sub(r"<think>[\s\S]*</think>", "", response.completion_text).strip()
                return generated_prompt

        return ""

    async def _call_sd_api(self, endpoint: str, payload: dict) -> dict:
        """通用API调用函数"""
        try:
            session = await self.ensure_session()
            async with session.post(
                    f"{self.config['webui_url']}{endpoint}",
                    json=payload
            ) as resp:
                if resp.status != 200:
                    error = (await resp.text())[:1000]
                    raise ConnectionError(f"API错误 ({resp.status}): {error}")
                return await resp.json()
        except asyncio.TimeoutError as e:
            raise TimeoutError("Stable Diffusion WebUI 请求超时") from e
        except aiohttp.ClientError as e:
            raise ConnectionError(f"连接失败: {e}") from e

    async def _call_t2i_api(self, prompt: str) -> dict:
        """调用 Stable Diffusion 文生图 API"""
        await self.ensure_session()
        payload = await self._generate_payload(prompt)
        return await self._call_sd_api("/sdapi/v1/txt2img", payload)

    async def _apply_image_processing(self, image_origin: str) -> str:
        """统一处理高分辨率修复与超分辨率放大"""

        # 获取配置参数
        params = self.config["default_params"]
        upscale_factor = params["upscale_factor"] or "2"
        upscaler = params["upscaler"] or "未设置"

        # 根据配置构建payload
        payload = {
            "image": image_origin,
            "upscaling_resize": upscale_factor,  # 使用配置的放大倍数
            "upscaler_1": upscaler,  # 使用配置的上采样算法
            "resize_mode": 0,  # 标准缩放模式
            "show_extras_results": True,  # 显示额外结果
            "upscaling_resize_w": 1,  # 自动计算宽度
            "upscaling_resize_h": 1,  # 自动计算高度
            "upscaling_crop": False,  # 不裁剪图像
            "gfpgan_visibility": 0,  # 不使用人脸修复
            "codeformer_visibility": 0,  # 不使用CodeFormer修复
            "codeformer_weight": 0,  # 不使用CodeFormer权重
            "extras_upscaler_2_visibility": 0  # 不使用额外的上采样算法
        }

        resp = await self._call_sd_api("/sdapi/v1/extra-single-image", payload)
        return resp["image"]

    async def _set_model(self, model_name: str) -> bool:
        """设置图像生成模型，并存入 config"""
        try:
            session = await self.ensure_session()
            async with session.post(
                    f"{self.config['webui_url']}/sdapi/v1/options",
                    json={"sd_model_checkpoint": model_name}
            ) as resp:
                if resp.status == 200:
                    self.config["base_model"] = model_name  # 存入 config
                    self.config.save_config()

                    logger.debug(f"模型已设置为: {model_name}")
                    return True
                else:
                    logger.error(f"设置模型失败 (状态码: {resp.status})")
                    return False
        except Exception as e:
            logger.error(f"设置模型异常: {e}")
            return False

    async def _check_webui_available(self) -> (bool, str):
        """服务状态检查"""
        try:
            await self.ensure_session()
            async with self.session.get(f"{self.config['webui_url']}/sdapi/v1/progress") as resp:
                if resp.status == 200:
                    return True, 0
                else:
                    logger.debug(f"⚠️ Stable diffusion Webui 返回值异常，状态码: {resp.status})")
                    return False, resp.status
        except Exception as e:
            logger.debug(f"❌ 测试连接 Stable diffusion Webui 失败，报错：{e}")
            return False, 0

    def _get_generation_params(self) -> str:
        """获取当前图像生成的参数"""
        global_positive_prompt_switch = self.config.get("global_prompt_group").get("global_positive_prompt_switch", False)  # 获取全局正面提示词开关状态
        global_negative_prompt_switch = self.config.get("global_prompt_group").get("global_negative_prompt_switch", False)  # 获取全局负面提示词开关状态
        global_positive_prompt = self.config.get("global_prompt_group").get("global_positive_prompt", "") # 获取全局正面提示词
        global_negative_prompt = self.config.get("global_prompt_group").get("global_negative_prompt", "")   #获取全局负面提示词

        params = self.config.get("default_params", {})
        width = params.get("width") or "未设置"
        height = params.get("height") or "未设置"
        steps = params.get("steps") or "未设置"
        sampler = params.get("sampler") or "未设置"
        cfg_scale = params.get("cfg_scale") or "未设置"
        batch_size = params.get("batch_size") or "未设置"
        n_iter = params.get("n_iter") or "未设置"

        base_model = self.config.get("base_model").strip() or "未设置"
        lora_tags = self._build_lora_tags()
        lora_display = ", ".join(lora_tags) if lora_tags else "未设置"

        return (
            f"- 全局正面提示词: {'开启' if global_positive_prompt_switch else '关闭'}\n"
            f"- 全局正面提示词: {global_positive_prompt}\n"
            f"- 全局负面提示词: {'开启' if global_negative_prompt_switch else '关闭'}\n"
            f"- 全局负面提示词: {global_negative_prompt}\n"
            f"- 上次插件设置模型: {base_model}\n"
            f"- 默认 LoRA: {lora_display}\n"
            f"- 图片尺寸: {width}x{height}\n"
            f"- 步数: {steps}\n"
            f"- 采样器: {sampler}\n"
            f"- CFG比例: {cfg_scale}\n"
            f"- 批数量: {batch_size}\n"
            f"- 迭代次数: {n_iter}"
        )

    def _get_upscale_params(self) -> str:
        """获取当前图像增强（超分辨率放大）参数"""
        params = self.config["default_params"]
        upscale_factor = params["upscale_factor"] or "2"
        upscaler = params["upscaler"] or "未设置"

        return (
            f"- 放大倍数: {upscale_factor}\n"
            f"- 上采样算法: {upscaler}"
        )

    @command_group("sd")
    def sd(self):
        pass

    @sd.command("check")    # 服务状态检查
    async def check(self, event: AstrMessageEvent):
        """服务状态检查"""
        try:
            webui_available, status = await self._check_webui_available()
            if webui_available:
                yield event.plain_result("✅ 同Webui连接正常")
            else:
                yield event.plain_result(f"❌ 同Webui无连接，请检查配置和Webui工作状态")
        except Exception as e:
            logger.error(f"❌ 检查可用性错误，报错{e}")
            yield event.plain_result("❌ 检查可用性错误，请检查日志")

    async def _run_generate_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        allow_generate_prompt: bool,
        allow_extract_prompt: bool,
        for_tool: bool = False
    ):
        """Shared image generation logic for command/tool callers.

        AstrBot 当前稳定版（v4.27.3）会包裹 `@llm_tool` 的异步生成器并只保留**最后**一个
        yield（`_PermissionGuardedTool.call`）。因此工具路径（for_tool=True）只 yield
        最终结果（图片），不 yield 任何过程提示/成功文字，否则图片会被"最后一条
        提示词"顶掉而丢失。
        """
        async with self.task_semaphore:
            self.active_tasks += 1
            try:
                if allow_extract_prompt:
                    prompt = self._extract_prompt_from_message(event, prompt)
                else:
                    prompt = (prompt or "").strip()
                if not prompt:
                    yield event.plain_result("⚠️ 需要提供提示词")
                    return
                # 检查webui可用性
                if not (await self._check_webui_available())[0]:
                    yield event.plain_result("⚠️ 同webui无连接，目前无法生成图片！")
                    return

                verbose = self.config["verbose"] and not for_tool
                if verbose:
                    yield event.plain_result("🖌️ 生成图像阶段，这可能需要一段时间...")

                # 生成正面提示词，决定到底是使用LLM生成还是用户直接提供
                generated_prompt = ""
                if allow_generate_prompt and self.config.get("enable_generate_prompt"):
                    generated_prompt = await self._generate_prompt(prompt)
                    logger.debug(f"LLM generated prompt: {generated_prompt}")

                positive_prompt = self._build_positive_prompt(prompt, generated_prompt)

                #输出正面提示词
                if self.config.get("enable_show_positive_prompt", False) and not for_tool:
                    yield event.plain_result(f"正面提示词：{positive_prompt}")

                # 生成图像
                response = await self._call_t2i_api(positive_prompt)
                if not response.get("images"):
                    raise ValueError("API返回数据异常：生成图像失败")

                images = response["images"]

                if len(images) == 1:

                    image_data = response["images"][0]

                    image = image_data

                    # 图像处理
                    if self.config.get("enable_upscale"):
                        if verbose:
                            yield event.plain_result("🖼️ 处理图像阶段，即将结束...")
                        image = await self._apply_image_processing(image)

                    yield event.chain_result([Image.fromBase64(image)])
                else:
                    chain = []

                    if self.config.get("enable_upscale") and verbose:
                        yield event.plain_result("🖼️ 处理图像阶段，即将结束...")

                    for image_data in images:
                        image = image_data

                        # 图像处理
                        if self.config.get("enable_upscale"):
                            image = await self._apply_image_processing(image)

                        # 添加到链对象
                        chain.append(Image.fromBase64(image))

                    # 将链式结果发送给事件
                    yield event.chain_result(chain)

                # 工具路径下不 yield 成功文字，确保图片是最后一个 yield
                if verbose:
                    yield event.plain_result("✅ 图像生成成功")

            except ValueError as e:
                # 针对API返回异常的处理
                logger.error(f"API返回数据异常: {e}")
                yield event.plain_result(f"❌ 图像生成失败: 参数异常，API调用失败")

            except ConnectionError as e:
                # 网络连接错误处理
                msg = str(e)
                logger.error(f"网络连接失败: {msg}")
                if "sampler" in msg.lower():
                    yield event.plain_result(
                        "⚠️ 生成失败: 采样器不兼容当前模型\n"
                        "请用 `/sd sampler list` 查看可用采样器，`/sd sampler set [索引]` 切换\n"
                        "提示: SD3/FLUX 通常需用 Euler/Euler a；SDXL 可用 DPM++ 2M Karras"
                    )
                else:
                    yield event.plain_result("⚠️ 生成失败! 请检查网络连接和WebUI服务是否运行正常")

            except TimeoutError as e:
                # 处理超时错误
                logger.error(f"请求超时: {e}")
                yield event.plain_result("⚠️ 请求超时，请稍后再试")

            except Exception as e:
                # 捕获所有其他异常
                logger.error(f"生成图像时发生其他错误: {e}")
                yield event.plain_result(f"❌ 图像生成失败: 发生其他错误，请检查日志")
            finally:
                self.active_tasks -= 1

    @sd.command("gen")  # 生成图像指令
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """生成图像指令
        Args:
            prompt: 图像描述提示词
        """
        async for result in self._run_generate_image(
            event,
            prompt,
            allow_generate_prompt=True,
            allow_extract_prompt=True
        ):
            yield result

    @sd.command("verbose")  # 切换详细输出模式
    async def set_verbose(self, event: AstrMessageEvent):
        """切换详细输出模式（verbose）"""
        try:
            # 读取当前状态并取反
            current_verbose = self.config.get("verbose", True)
            new_verbose = not current_verbose

            # 更新配置
            self.config["verbose"] = new_verbose
            self.config.save_config()

            # 发送反馈消息
            status = "开启" if new_verbose else "关闭"
            yield event.plain_result(f"📢 详细输出模式已{status}")
        except Exception as e:
            logger.error(f"切换详细输出模式失败: {e}")
            yield event.plain_result("❌ 切换详细模式失败，请检查日志")

    @sd.command("upscale") # 切换图像增强模式
    async def set_upscale(self, event: AstrMessageEvent):
        """设置图像增强模式（enable_upscale）"""
        try:
            # 获取当前的 upscale 配置值
            current_upscale = self.config.get("enable_upscale", False)

            # 切换 enable_upscale 配置
            new_upscale = not current_upscale

            # 更新配置
            self.config["enable_upscale"] = new_upscale
            self.config.save_config()

            # 发送反馈消息
            status = "开启" if new_upscale else "关闭"
            yield event.plain_result(f"📢 图像增强模式已{status}")

        except Exception as e:
            logger.error(f"切换图像增强模式失败: {e}")
            yield event.plain_result("❌ 切换图像增强模式失败，请检查日志")

    @sd.command("LLM")  # 切换生成提示词功能
    async def set_generate_prompt(self, event: AstrMessageEvent):
        """切换生成提示词功能"""
        try:
            current_setting = self.config.get("enable_generate_prompt", False)
            new_setting = not current_setting
            self.config["enable_generate_prompt"] = new_setting
            self.config.save_config()

            status = "开启" if new_setting else "关闭"
            yield event.plain_result(f"📢 提示词生成功能已{status}")
        except Exception as e:
            logger.error(f"切换生成提示词功能失败: {e}")
            yield event.plain_result("❌ 切换生成提示词功能失败，请检查日志")

    @sd.command("headtail") # 切换全局正面提示词添加位置
    async def switch_positive_prompt_add_in_head_or_tail(self, event: AstrMessageEvent):
        """切换全局正面提示词添加位置"""
        try:
            current_setting = self.config.get("global_prompt_group").get("positive_prompt_add_in_head_or_tail_switch", False)
            new_setting = not current_setting
            self.config["global_prompt_group"]["positive_prompt_add_in_head_or_tail_switch"] = new_setting
            self.config.save_config()

            status = "头部" if new_setting else "尾部"
            yield event.plain_result(f"📢 全局正面提示词现将添加在 {status}")
        except Exception as e:
            logger.error(f"切换全局正面提示词位置失败: {e}")
            yield event.plain_result("❌ 切换全局正面提示词位置失败，请检查日志")

    @sd.command("prompt") # 切换显示正面提示词功能
    async def set_show_prompt(self, event: AstrMessageEvent):
        """切换显示正面提示词功能"""
        try:
            current_setting = self.config.get("enable_show_positive_prompt", False)
            new_setting = not current_setting
            self.config["enable_show_positive_prompt"] = new_setting
            self.config.save_config()

            status = "开启" if new_setting else "关闭"
            yield event.plain_result(f"📢 显示正面提示词功能已{status}")
        except Exception as e:
            logger.error(f"切换显示正面提示词功能失败: {e}")
            yield event.plain_result("❌ 切换显示正面提示词功能失败，请检查日志")

    @sd.command("vocab")  # 查看标准词库状态 / 检索预览
    async def show_vocab(self, event: AstrMessageEvent, query: str = ""):
        """查看标准词库状态；附带查询词时可预览向量检索结果"""
        try:
            path = (self.config.get("prompt_vocabulary_path") or "").strip()
            if not path:
                yield event.plain_result("⚠️ 未配置标准词库文件路径")
                return
            abs_path = path if os.path.isabs(path) else os.path.abspath(path)
            entries = self._get_vocab_index()
            if not entries:
                yield event.plain_result(f"⚠️ 标准词库未加载\n路径: {abs_path}\n请检查文件是否存在或为空")
                return
            nonempty = sum(1 for t, c in entries if c)
            embed_line = self._embedding_status_line()
            header = (
                f"📄 标准词库路径: {abs_path}\n"
                f"条目总数: {len(entries)}（其中有内容 {nonempty} 条）\n"
                f"Top-K: {self.config.get('prompt_vocabulary_top_k', 8)}  "
                f"最大注入字数: {self.config.get('prompt_vocabulary_max_chars', 4000)}\n"
                f"{embed_line}"
            )
            if not query.strip():
                yield event.plain_result(header)
                return
            if self._embed_state != "ready":
                yield event.plain_result(f"{header}\n\n⚠️ 向量检索未就绪，无法预览命中片段")
                return
            snippet = await self._retrieve_vocab(query)
            if not snippet:
                yield event.plain_result(f"{header}\n\n检索「{query}」：无命中片段")
            else:
                preview = snippet if len(snippet) <= 800 else snippet[:800] + "\n...(已截断)"
                yield event.plain_result(f"{header}\n\n检索「{query}」命中片段预览:\n{preview}")
        except Exception as e:
            logger.error(f"查看标准词库失败: {e}")
            yield event.plain_result("❌ 查看标准词库失败，请检查日志")

    def _embedding_status_line(self) -> str:
        """生成 embedding 检索状态摘要（用于 /sd vocab 与 /sd embedding status）"""
        if not self.config.get("embedding_enabled", True):
            return "🧠 向量检索: 已关闭（embedding_enabled=false）"
        state_desc = {
            "idle": "未启用（未配置词库）",
            "building": "索引构建中…",
            "ready": "",
            "error": f"不可用（{self._embed_error}）" if self._embed_error else "不可用",
        }.get(self._embed_state, self._embed_state)
        if self._embed_state == "ready" and self._embed_data:
            state_desc = f"就绪（{self._embed_data['model']} / {self._embed_data['dim']} 维 / {len(self._embed_data['entries'])} 条）"
        return f"🧠 向量检索: {state_desc}"

    @sd.group("embedding")  # 词库向量检索（embedding）子命令
    def embedding(self):
        pass

    @embedding.command("status")  # 查看 embedding 检索状态
    async def embedding_status(self, event: AstrMessageEvent):
        """查看词库向量检索（embedding）状态"""
        try:
            provider, api_base, _ = self._get_embedding_provider()
            if provider is None:
                yield event.plain_result("🧠 未找到可用的 embedding 提供商\n在 AstrBot 提供商管理中配置 Embedding 提供商后，本插件将自动使用其模型。")
                return
            pcfg = getattr(provider, "provider_config", None) or {}
            provider_name = pcfg.get("name") or pcfg.get("id") or str(pcfg.get("type") or "未知")
            if hasattr(provider, "get_embeddings"):
                # 原生 EmbeddingProvider：模型完全由 AstrBot 提供商配置决定
                model = self._resolve_embedding_model(provider)
                lines = [
                    f"🧠 提供商: {provider_name}（AstrBot Embedding 提供商，类型 {pcfg.get('type')}）",
                    f"🔗 API 地址: {pcfg.get('embedding_api_base') or '由 AstrBot 统一管理'}",
                    f"📦 Embedding 模型: {model or '未知'}",
                ]
            else:
                # OpenAI 兼容回退路径
                model = self._resolve_embedding_model(provider)
                lines = [
                    f"🧠 提供商: {provider_name}（对话提供商 OpenAI 兼容回退）",
                    f"🔗 API 地址: {api_base or '未配置'}",
                    f"📦 Embedding 模型: {model or '自动探测中（建议手动指定）'}",
                ]
            data = self._embed_data or {}
            lines += [
                f"📚 词库条目: {len(data.get('entries', []))}",
                f"📐 向量维度: {data.get('dim', '-')}",
                f"⚙️ 状态: {self._embedding_status_line().replace('🧠 向量检索: ', '')}",
            ]
            yield event.plain_result("\n".join(lines))
        except Exception as e:
            logger.error(f"查看 embedding 状态失败: {e}")
            yield event.plain_result("❌ 查看 embedding 状态失败，请检查日志")

    @embedding.command("provider")  # 查看/切换 embedding 提供商
    async def embedding_provider_switch(self, event: AstrMessageEvent, provider_id: str = ""):
        """列出 AstrBot 已配置的 Embedding 提供商；传入 ID 可切换（如 /sd embedding provider <id>）"""
        try:
            get_all_eps = getattr(self.context, "get_all_embedding_providers", None)
            eps = get_all_eps() if get_all_eps is not None else []
            current = (self.config.get("embedding_provider_id") or "").strip()
            if provider_id:
                get_by_id = getattr(self.context, "get_provider_by_id", None) or getattr(
                    self.context, "get_provider", None
                )
                target = None
                if get_by_id is not None:
                    try:
                        target = get_by_id(provider_id)
                    except Exception:
                        target = None
                if target is None and get_all_eps is not None:
                    for p in eps:
                        if (getattr(p, "provider_config", None) or {}).get("id") == provider_id:
                            target = p
                            break
                if target is None:
                    yield event.plain_result(f"❌ 未找到提供商: {provider_id}\n可用: /sd embedding provider 查看列表")
                    return
                self.config["embedding_provider_id"] = provider_id
                self.config.save_config()
                # 取消旧构建任务，重置状态，用新提供商后台重建（provider_key 变化自动触发）
                self._cancel_embedding_build()
                self._embed_state = "idle"
                self._embed_data = None
                self._embed_error = None
                try:
                    self._embed_build_task = asyncio.get_running_loop().create_task(
                        self._ensure_embedding_index(force=False)
                    )
                except Exception:
                    pass
                yield event.plain_result(f"✅ 已切换到提供商: {provider_id}\n索引将在后台重建，可用 /sd embedding status 查看进度")
                return
            lines = ["📋 AstrBot 已配置的 Embedding 提供商:"]
            if not eps:
                lines.append("（无）— 请在 AstrBot 提供商管理中添加 Embedding 提供商")
            for p in eps:
                pcfg = getattr(p, "provider_config", None) or {}
                model = pcfg.get("embedding_model") or getattr(p, "model_name", "") or "未知"
                lines.append(f"- {pcfg.get('id')}: {model}（类型 {pcfg.get('type')}）")
            lines.append(f"当前选择: {current or '（自动：第一个可用）'}")
            lines.append("用法: /sd embedding provider <id>")
            yield event.plain_result("\n".join(lines))
        except Exception as e:
            logger.error(f"操作 embedding 提供商失败: {e}")
            yield event.plain_result(f"❌ 操作失败: {e}")

    @embedding.command("rebuild")  # 强制重建词库向量索引
    async def embedding_rebuild(self, event: AstrMessageEvent):
        """强制重建词库向量索引（词库文件或模型变更后会自动重建，一般无需手动）"""
        try:
            self._cancel_embedding_build()
            self._embed_state = "idle"
            self._embed_data = None
            self._embed_build_task = asyncio.get_running_loop().create_task(
                self._ensure_embedding_index(force=True)
            )
            yield event.plain_result("🔨 已开始重建词库向量索引，完成后可用 /sd embedding status 查看")
        except Exception as e:
            logger.error(f"启动重建失败: {e}")
            yield event.plain_result(f"❌ 无法启动重建: {e}")

    @sd.command("timeout")  # 设置会话超时时间
    async def set_timeout(self, event: AstrMessageEvent, time: int):
        """设置会话超时时间"""
        try:
            if time < 10 or time > 1800:
                yield event.plain_result("⚠️ 超时时间需设置在 10秒 到 1800秒 范围内")
                return

            self.config["session_timeout_time"] = time
            self.config.save_config()

            yield event.plain_result(f"⏲️ 会话超时时间已设置为 {time} 秒")
        except Exception as e:
            logger.error(f"设置会话超时时间失败: {e}")
            yield event.plain_result("❌ 设置会话超时时间失败，请检查日志")

    @sd.command("conf") # 输出当前各项配置
    async def show_conf(self, event: AstrMessageEvent):
        """打印当前图像生成参数，包括当前使用的模型"""
        try:
            global_positive_prompt_switch = self.config.get("global_prompt_group").get("global_positive_prompt_switch", False)  # 获取全局正面提示词开关状态
            global_negative_prompt_switch = self.config.get("global_prompt_group").get("global_negative_prompt_switch", False)  # 获取全局负面提示词开关状态

            gen_params = self._get_generation_params()  # 获取当前图像参数
            scale_params = self._get_upscale_params()   # 获取图像增强参数
            prompt_guidelines = self.config.get("prompt_guidelines").strip() or "未设置"  # 获取提示词限制

            prompt_vocabulary_path = (self.config.get("prompt_vocabulary_path") or "").strip() or "未设置"
            vocab_entries = self._get_vocab_index()
            vocab_status = f"已加载 ({len(vocab_entries)} 条目)" if vocab_entries else "未加载"

            new_params = self.config.get("new_model_params", {})
            clip_skip = new_params.get("clip_skip", 0)
            refiner_ckpt = (new_params.get("refiner_checkpoint") or "").strip() or "未设置"
            refiner_switch = new_params.get("refiner_switch_at", 0.8)
            current_model, current_vae = await self._get_current_webui_model_info()

            positive_prompt_add_in_head_or_tail_switch = self.config.get("global_prompt_group").get('positive_prompt_add_in_head_or_tail_switch',False) # 获取全局正面提示词添加位置

            verbose = self.config.get("verbose", True)  # 获取详略模式
            upscale = self.config.get("enable_upscale", False)  # 图像增强模式
            show_positive_prompt = self.config.get("enable_show_positive_prompt", False)  # 是否显示正面提示词
            generate_prompt = self.config.get("enable_generate_prompt", False)  # 是否启用生成提示词

            conf_message = (
                f"⚙️  图像生成参数:\n{gen_params}\n\n"
                f"🆕  新模型参数:\n"
                f"- WebUI当前模型: {current_model}\n"
                f"- WebUI当前VAE: {current_vae}\n"
                f"- CLIP跳过: {clip_skip}\n"
                f"- Refiner模型: {refiner_ckpt}\n"
                f"- Refiner切换: {refiner_switch}\n\n"
                f"⬅️➡️  全局正面提示词加在 {'头部' if positive_prompt_add_in_head_or_tail_switch else '尾部'}\n\n"
                f"🔍  图像增强参数:\n{scale_params}\n\n"
                f"🛠️  提示词附加要求: {prompt_guidelines}\n\n"
                f"📚  标准词库路径: {prompt_vocabulary_path}\n\n"
                f"📚  标准词库状态: {vocab_status}\n\n"
                f"📢  详细输出模式: {'开启' if verbose else '关闭'}\n\n"
                f"🔧  图像增强模式: {'开启' if upscale else '关闭'}\n\n"
                f"📝  正面提示词显示: {'开启' if show_positive_prompt else '关闭'}\n\n"
                f"🤖  提示词生成模式: {'开启' if generate_prompt else '关闭'}"
            )

            yield event.plain_result(conf_message)
        except Exception as e:
            logger.error(f"获取生成参数失败: {e}")
            yield event.plain_result("❌ 获取图像生成参数失败，请检查配置是否正确")

    @sd.command("help") # 帮助指令
    async def show_help(self, event: AstrMessageEvent):
        """显示SDGenerator插件所有可用指令及其描述"""
        help_msg = [
            "🖼️ **Stable Diffusion 插件帮助指南**",
            "该插件用于调用 Stable Diffusion WebUI 的 API 生成图像并管理相关模型资源。",
            "",
            "📜 **主要功能指令**:",
            "- `/sd gen [提示词]`：生成图片，例如 `/sd gen 星空下的城堡`。",
            "- `/sd check`：检查 WebUI 的连接状态。",
            "- `/sd conf`：显示当前使用配置，包括模型、参数和提示词设置。",
            "- `/sd help`：显示本帮助信息。",
            "",
            "➕➖ **正负提示词设置指令**:",
            "- `/sd headtail`：切换全局正面提示词添加位置（头部或尾部）。",
            "",
            "🔧 **高级功能指令**:",
            "- `/sd verbose`：切换详细输出模式，用于实时告知目前AI生图进行到了哪个阶段。",
            "- `/sd upscale`：切换图像增强模式（用于超分辨率放大或高分修复）。",
            "- `/sd LLM`：开启后，在使用/sd gen指令时，将内容先发送给LLM，再由LLM来生成正面提示词",
            "- `/sd prompt`：开启时，用户发起AI生图请求后，将发送一条消息，内容为送入到Stable diffusion的正面提示词",
            "- `/sd vocab`：查看标准词库状态；附带描述（如 `/sd vocab 沙滩泳装`）可预览 embedding 语义检索命中片段。",
            "- `/sd embedding status`：查看词库向量索引状态。",
            "- `/sd embedding provider [ID]`：列出或切换 AstrBot Embedding 提供商。",
            "- `/sd embedding rebuild`：在后台强制重建词库向量索引。",
            "- `/sd timeout [秒数]`：设置连接超时时间（建议范围：10 到 1800 秒）。",
            "- `/sd res  [宽度] [高度]`：设置图像生成的分辨率（高度和宽度均支持:1-2048之间的任意整数）。",
            "- `/sd step [步数]`：设置图像生成的步数（范围：10 到 50 步）。",
            "- `/sd batch [数量]`：设置发出AI生图请求后，每轮生成的图片数量（范围： 1 到 10 张）。"
            "- `/sd iter [次数]`：设置迭代次数（范围： 1 到 5 次）。",
            "",
            "🆕 **新模型支持指令**:",
            "- `/sd preset [名称]`：一键切换模型预设（sd15/sdxl/sd3/flux/pony/noobai/illustrious），自动设置分辨率/步数/CFG/clip_skip。",
            "- `/sd clipskip [层数]`：设置 CLIP 跳过层数（0-12）。动漫新模型(Pony/NoobAI/Illustrious)通常设2。",
            "- `/sd refiner [索引]`：设置 SDXL Refiner 精修模型（索引同 /sd model list），0 清除。",
            "",
            "🖼️ **基本模型与微调模型指令**:",
            "- `/sd model list`：列出 WebUI 当前可用的模型。",
            "- `/sd model set [索引]`：利用索引设置模型，索引可通过 `model list` 查询。",
            "- `/sd lora list`：列出所有可用的 LoRA 模型及当前默认 LoRA。",
            "- `/sd lora set [名字] [权重]`：设置默认 LoRA（如 `/sd lora set chibi 0.8`，可多次设置多个，生图自动带上）。",
            "- `/sd lora clear`：清空全部默认 LoRA。",
            "- `/sd embedding`：显示所有已加载的 Embedding 模型。",
            "",
            "🎨 **采样器与上采样算法指令**:",
            "- `/sd sampler list`：列出支持的采样器。",
            "- `/sd sampler set [索引]`：根据索引配置采样器，用于调整生成效果。",
            "- `/sd upscaler list`：列出支持的上采样算法。",
            "- `/sd upscaler set [索引]`：根据索引设置上采样算法。",
            "",
            "ℹ️ **注意事项**:",
            "- 如启用自动生成提示词功能，则会使用 LLM 利用提供的内容来生成提示词。",
            "- 提示词可以直接包含空格、英文逗号和 Danbooru tags，无需使用特殊字符替代空格。",
            "- 模型、采样器和其他资源的索引需要使用对应 `list` 命令获取后设置！",
        ]
        yield event.plain_result("\n".join(help_msg))

    @sd.command("res") # 设置生成图像的宽和高
    async def set_resolution(self, event: AstrMessageEvent, width: int,height: int ):
        """设置分辨率"""
        try:
            if not isinstance(height, int) or not isinstance(width, int) or height < 1 or width < 1 or height > 2048 or width > 2048:
                yield event.plain_result("⚠️ 分辨率仅支持:1-2048之间的任意整数")
                return

            self.config["default_params"]["height"] = height
            self.config["default_params"]["width"] = width
            self.config.save_config()

            yield event.plain_result(f"✅ 图像生成的分辨率已设置为: 宽度——{width}，高度——{height}")
        except Exception as e:
            logger.error(f"设置分辨率失败: {e}")
            yield event.plain_result("❌ 设置分辨率失败，请检查日志")

    @sd.command("step")# 设置生成图像的步数
    async def set_step(self, event: AstrMessageEvent, step: int):
        """设置步数"""
        try:
            if step < 10 or step > 50:
                yield event.plain_result("⚠️ 步数需设置在 10 到 50 之间")
                return

            self.config["default_params"]["steps"] = step
            self.config.save_config()

            yield event.plain_result(f"✅ 步数已设置为: {step}")
        except Exception as e:
            logger.error(f"设置步数失败: {e}")
            yield event.plain_result("❌ 设置步数失败，请检查日志")

    @sd.command("batch") # 设置一次性生成的图片数量
    async def set_batch_size(self, event: AstrMessageEvent, batch_size: int):
        """设置批量生成的图片数量"""
        try:
            if batch_size < 1 or batch_size > 10:
                yield event.plain_result("⚠️ 图片生成的批数量需设置在 1 到 10 之间")
                return

            self.config["default_params"]["batch_size"] = batch_size
            self.config.save_config()

            yield event.plain_result(f"✅ 图片生成批数量已设置为: {batch_size}")
        except Exception as e:
            logger.error(f"设置批量生成数量失败: {e}")
            yield event.plain_result("❌ 设置图片生成批数量失败，请检查日志")

    @sd.command("iter") # 设置生成图像的迭代次数
    async def set_n_iter(self, event: AstrMessageEvent, n_iter: int):
        """设置生成迭代次数"""
        try:
            if n_iter < 1 or n_iter > 5:
                yield event.plain_result("⚠️ 图片生成的迭代次数需设置在 1 到 5 之间")
                return

            self.config["default_params"]["n_iter"] = n_iter
            self.config.save_config()

            yield event.plain_result(f"✅ 图片生成的迭代次数已设置为: {n_iter}")
        except Exception as e:
            logger.error(f"设置生成迭代次数失败: {e}")
            yield event.plain_result("❌ 设置图片生成的迭代次数失败，请检查日志")

    # 新模型参数快捷指令
    MODEL_PRESETS = {
        "sd15": {"width": 512, "height": 512, "steps": 20, "cfg_scale": 7.0, "clip_skip": 0},
        "sdxl": {"width": 1024, "height": 1024, "steps": 30, "cfg_scale": 7.0, "clip_skip": 2},
        "sd3": {"width": 1024, "height": 1024, "steps": 30, "cfg_scale": 5.0, "clip_skip": 0},
        "flux": {"width": 1024, "height": 1024, "steps": 20, "cfg_scale": 3.5, "clip_skip": 0},
        "pony": {"width": 1024, "height": 1024, "steps": 25, "cfg_scale": 7.0, "clip_skip": 2},
        "noobai": {"width": 1024, "height": 1024, "steps": 30, "cfg_scale": 7.0, "clip_skip": 2},
        "illustrious": {"width": 1024, "height": 1024, "steps": 30, "cfg_scale": 7.0, "clip_skip": 2},
    }

    @sd.command("clipskip")  # 设置 CLIP 跳过层数
    async def set_clip_skip(self, event: AstrMessageEvent, clip_skip: int):
        """设置 CLIP 跳过层数（新模型关键参数）"""
        try:
            if clip_skip < 0 or clip_skip > 12:
                yield event.plain_result("⚠️ CLIP 跳过层数需设置在 0 到 12 之间")
                return

            self.config.setdefault("new_model_params", {})["clip_skip"] = clip_skip
            self.config.save_config()

            yield event.plain_result(f"✅ CLIP 跳过层数已设置为: {clip_skip}")
        except Exception as e:
            logger.error(f"设置 CLIP 跳过层数失败: {e}")
            yield event.plain_result("❌ 设置 CLIP 跳过层数失败，请检查日志")

    @sd.command("refiner")  # 设置 SDXL Refiner 精修模型
    async def set_refiner(self, event: AstrMessageEvent, model_index: int):
        """设置 SDXL Refiner 精修模型，0 清除"""
        try:
            if model_index == 0:
                self.config.setdefault("new_model_params", {})["refiner_checkpoint"] = ""
                self.config.save_config()
                yield event.plain_result("✅ 已清除 Refiner 精修模型")
                return

            models = await self._get_sd_model_list()
            if not models:
                yield event.plain_result("⚠️ 没有可用的模型")
                return

            index = model_index - 1
            if index < 0 or index >= len(models):
                yield event.plain_result("❌ 无效的模型索引，请使用 /sd model list 获取")
                return

            selected = models[index]
            self.config.setdefault("new_model_params", {})["refiner_checkpoint"] = selected
            self.config.save_config()
            yield event.plain_result(f"✅ Refiner 精修模型已设置为: {selected}")
        except Exception as e:
            logger.error(f"设置 Refiner 模型失败: {e}")
            yield event.plain_result("❌ 设置 Refiner 模型失败，请检查日志")

    @sd.command("preset")  # 一键切换模型预设
    async def set_preset(self, event: AstrMessageEvent, name: str):
        """一键切换新模型预设（分辨率/步数/CFG/clip_skip）"""
        try:
            key = (name or "").strip().lower()
            if key not in self.MODEL_PRESETS:
                available = ", ".join(self.MODEL_PRESETS.keys())
                yield event.plain_result(f"⚠️ 未知预设: {name}\n可用预设: {available}")
                return

            p = self.MODEL_PRESETS[key]
            dp = self.config["default_params"]
            dp["width"] = p["width"]
            dp["height"] = p["height"]
            dp["steps"] = p["steps"]
            dp["cfg_scale"] = p["cfg_scale"]
            self.config.setdefault("new_model_params", {})["clip_skip"] = p["clip_skip"]
            self.config.save_config()

            yield event.plain_result(
                f"✅ 已切换到 {key} 预设:\n"
                f"- 分辨率: {p['width']}x{p['height']}\n"
                f"- 步数: {p['steps']}\n"
                f"- CFG: {p['cfg_scale']}\n"
                f"- CLIP跳过: {p['clip_skip']}\n"
                f"⚠️ 请确认采样器兼容该模型（SD3/FLUX 需用 Euler 等采样器，可用 /sd sampler list 查看）"
            )
        except Exception as e:
            logger.error(f"切换预设失败: {e}")
            yield event.plain_result("❌ 切换预设失败，请检查日志")

    @sd.group("model") #引出模型设置子命令
    def model(self):
        pass

    @model.command("list") # 列出可用的生图模型
    async def list_model(self, event: AstrMessageEvent):
        """
        以“1. xxx.safetensors“形式打印可用的模型
        """
        try:
            models = await self._get_sd_model_list()  # 使用统一方法获取模型列表
            if not models:
                yield event.plain_result("⚠️ 没有可用的模型")
                return

            model_list = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(models))
            yield event.plain_result(f"🖼️ 可用模型列表:\n{model_list}")

        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            yield event.plain_result("❌ 获取模型列表失败，请检查 WebUI 是否运行")

    @model.command("set") # 设置使用哪个生图模型
    async def set_base_model(self, event: AstrMessageEvent, model_index: int):
        """
        解析用户输入的索引，并设置对应的模型
        """
        try:
            models = await self._get_sd_model_list()
            if not models:
                yield event.plain_result("⚠️ 没有可用的模型")
                return

            try:
                index = int(model_index) - 1  # 转换为 0-based 索引
                if index < 0 or index >= len(models):
                    yield event.plain_result("❌ 无效的模型索引，请使用 /sd model list 获取")
                    return

                selected_model = models[index]
                logger.debug(f"selected_model: {selected_model}")
                if await self._set_model(selected_model):
                    yield event.plain_result(f"✅ 模型已切换为: {selected_model}")
                else:
                    yield event.plain_result("⚠️ 切换模型失败，请检查 WebUI 状态")

            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")

        except Exception as e:
            logger.error(f"切换模型失败: {e}")
            yield event.plain_result("❌ 切换模型失败，请检查日志")

    @sd.group("lora")  # LoRA 设置子命令：list / set / clear
    def lora(self):
        pass

    @lora.command("list")  # 列出可用的 LoRA 模型
    async def list_lora(self, event: AstrMessageEvent):
        """
        列出可用的 LoRA 模型及当前默认 LoRA
        """
        try:
            lora_models = await self._get_lora_list()
            current = self._build_lora_tags()
            current_line = (
                f"\n\n⭐ 当前默认 LoRA: {', '.join(current)}"
                if current
                else "\n\n⭐ 当前未设置默认 LoRA（可用 /sd lora set <名字> [权重] 设置）"
            )
            if not lora_models:
                yield event.plain_result("没有可用的 LoRA 模型。")
            else:
                lora_model_list = "\n".join(f"{i + 1}. {lora}" for i, lora in enumerate(lora_models))
                yield event.plain_result(f"可用的 LoRA 模型:\n{lora_model_list}{current_line}")
        except Exception as e:
            yield event.plain_result(f"获取 LoRA 模型列表失败: {str(e)}")

    @lora.command("set")  # 设置默认 LoRA
    async def set_lora(self, event: AstrMessageEvent, lora_name: str, weight: float = 1.0):
        """
        设置默认 LoRA（写入配置，生图时自动带上；重复设置会覆盖同名 LoRA，不同名则追加）
        用法: /sd lora set <名字> [权重]，如 /sd lora set chibi 0.8
        """
        try:
            lora_name = (lora_name or "").strip()
            if not lora_name:
                yield event.plain_result("❌ 用法: /sd lora set <名字> [权重]，如 /sd lora set chibi 0.8")
                return

            # 与 /sd lora list 显示的名字对齐（WebUI 返回 name 字段）
            try:
                lora_models = await self._get_lora_list()
                matched = None
                for m in lora_models:
                    if m == lora_name or (m and lora_name.lower() in m.lower()):
                        matched = m
                        break
                if matched:
                    lora_name = matched
            except Exception:
                pass  # 列表获取失败时直接信任用户输入

            if weight <= 0:
                weight = 1.0

            new_params = self.config.setdefault("new_model_params", {})
            raw = (new_params.get("lora") or "").strip()
            items = [i.strip() for i in raw.split(",") if i.strip()]

            # 同名覆盖，不同名追加
            new_item = f"{lora_name}:{weight}"
            items = [new_item if i.split(":")[0].strip() == lora_name else i for i in items]
            if new_item not in items:
                items.append(new_item)

            new_params["lora"] = ", ".join(items)
            self.config.save_config()
            yield event.plain_result(f"✅ 已设置默认 LoRA: {new_item}\n当前全部: {', '.join(items)}")
        except Exception as e:
            logger.error(f"设置 LoRA 失败: {e}")
            yield event.plain_result(f"❌ 设置 LoRA 失败: {str(e)}")

    @lora.command("clear")  # 清空默认 LoRA
    async def clear_lora(self, event: AstrMessageEvent):
        """
        清空全部默认 LoRA
        """
        try:
            new_params = self.config.setdefault("new_model_params", {})
            if new_params.get("lora"):
                new_params["lora"] = ""
                self.config.save_config()
                yield event.plain_result("✅ 已清空全部默认 LoRA")
            else:
                yield event.plain_result("ℹ️ 当前本来就没有设置默认 LoRA")
        except Exception as e:
            logger.error(f"清空 LoRA 失败: {e}")
            yield event.plain_result(f"❌ 清空 LoRA 失败: {str(e)}")

    @sd.group("sampler") # 引出采样器设置子命令
    def sampler(self):
        pass

    @sampler.command("list") # 列出可用的采样器
    async def list_sampler(self, event: AstrMessageEvent):
        """
        列出所有可用的采样器
        """
        try:
            samplers = await self._get_sampler_list()
            if not samplers:
                yield event.plain_result("⚠️ 没有可用的采样器")
                return

            sampler_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(samplers))
            yield event.plain_result(f"🖌️ 可用采样器列表:\n{sampler_list}")
        except Exception as e:
            yield event.plain_result(f"获取采样器列表失败: {str(e)}")

    @sampler.command("set") # 设置采样器
    async def set_sampler(self, event: AstrMessageEvent, sampler_index: int):
        """
        设置采样器
        """
        try:
            samplers = await self._get_sampler_list()
            if not samplers:
                yield event.plain_result("⚠️ 没有可用的采样器")
                return

            try:
                index = int(sampler_index) - 1
                if index < 0 or index >= len(samplers):
                    yield event.plain_result("❌ 无效的采样器索引，请使用 /sd sampler list 获取")
                    return

                selected_sampler = samplers[index]
                self.config["default_params"]["sampler"] = selected_sampler
                self.config.save_config()

                yield event.plain_result(f"✅ 已设置采样器为: {selected_sampler}")
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")
        except Exception as e:
            yield event.plain_result(f"设置采样器失败: {str(e)}")

    @sd.group("upscaler") # 引出上采样算法设置子命令
    def upscaler(self):
        pass

    @upscaler.command("list")
    async def list_upscaler(self, event: AstrMessageEvent):
        """
        列出所有可用的上采样算法
        """
        try:
            upscalers = await self._get_upscaler_list()
            if not upscalers:
                yield event.plain_result("⚠️ 没有可用的上采样算法")
                return

            upscaler_list = "\n".join(f"{i + 1}. {u}" for i, u in enumerate(upscalers))
            yield event.plain_result(f"🖌️ 可用上采样算法列表:\n{upscaler_list}")
        except Exception as e:
            yield event.plain_result(f"获取上采样算法列表失败: {str(e)}")

    @upscaler.command("set") # 设置上采样算法
    async def set_upscaler(self, event: AstrMessageEvent, upscaler_index: int):
        """
        设置上采样算法
        """
        try:
            upscalers = await self._get_upscaler_list()
            if not upscalers:
                yield event.plain_result("⚠️ 没有可用的上采样算法")
                return

            try:
                index = int(upscaler_index) - 1
                if index < 0 or index >= len(upscalers):
                    yield event.plain_result("❌ 无效的上采样算法索引，请检查 /sd upscaler list")
                    return

                selected_upscaler = upscalers[index]
                self.config["default_params"]["upscaler"] = selected_upscaler
                self.config.save_config()

                yield event.plain_result(f"✅ 已设置上采样算法为: {selected_upscaler}")
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")
        except Exception as e:
            yield event.plain_result(f"设置上采样算法失败: {str(e)}")


    @sd.command("embedding") # 列出可用的 Embedding 模型
    async def list_embedding(self, event: AstrMessageEvent):
        """
        列出可用的 Embedding 模型
        """
        try:
            embedding_models = await self._get_embedding_list()
            if not embedding_models:
                yield event.plain_result("没有可用的 Embedding 模型。")
            else:
                embedding_model_list = "\n".join(f"{i + 1}. {lora}" for i, lora in enumerate(embedding_models))
                yield event.plain_result(f"可用的 Embedding 模型:\n{embedding_model_list}")
        except Exception as e:
            yield event.plain_result(f"获取 Embedding 模型列表失败: {str(e)}")

    @llm_tool("generate_image") # LLM可调用的图像生成工具函数
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str):
        """Generate an image using Stable Diffusion based on the given prompt.

        Call this tool whenever the user explicitly asks to generate, draw, create,
        or produce an image or illustration (e.g. "画一张", "生成图片", "draw a girl",
        "make an anime cover"). Do NOT call it for image searching, viewing, or uploading.

        The prompt must describe only the desired visible content. Use concise English,
        comma-separated Danbooru-style tags covering subject, appearance, clothing, pose,
        composition, environment, lighting, color and visual style. Keep character names
        and franchise names when relevant. Do not include chat commentary, tool instructions,
        image dimensions, sampler settings, LoRA syntax or negative prompts; those are managed
        by the plugin configuration.

        Args:
            prompt (string): English comma-separated tags describing the requested image.
        """
        try:
            # 使用 async for 遍历异步生成器的返回值
            async for result in self._run_generate_image(
                event,
                prompt,
                allow_generate_prompt=False,
                allow_extract_prompt=False,
                for_tool=True
            ):
                # 根据生成器的每一个结果返回响应
                yield result

        except Exception as e:
            logger.error(f"调用 generate_image 时出错: {e}")
            yield event.plain_result("❌ 图像生成失败，请检查日志")
