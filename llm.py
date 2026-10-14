import time
import os
import json
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

DEFAULT_SYSTEM_PROMPT = '你是一个知识助手，尽量以简短、口语化的方式输出'

# 知识库缓存：path -> 解析后的 dict（None 表示加载失败/不存在）
_kb_cache: dict = {}


def _load_kb(path: str):
    """加载公司知识库 JSON（带进程级缓存）。无路径/不存在/解析失败均返回 None。"""
    if not path:
        return None
    if path in _kb_cache:
        return _kb_cache[path]
    kb = None
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                kb = json.load(f)
            logger.info(f"[KB] loaded knowledge base: {path}")
        else:
            logger.warning(f"[KB] knowledge base file not found: {path}")
    except Exception as e:
        logger.exception(f"[KB] failed to load {path}: {e}")
        kb = None
    _kb_cache[path] = kb
    return kb


def _build_system_prompt(opt) -> str:
    """根据 --kb_path 知识库拼装 system prompt；未配置则回退通用提示。"""
    kb = _load_kb(getattr(opt, 'kb_path', '') or '')
    if not kb:
        return DEFAULT_SYSTEM_PROMPT
    lines = [kb.get('persona', '你是公司迎宾数字人助手。')]
    facts = kb.get('facts', {}) or {}
    if facts:
        lines.append('\n【公司资料】')
        for k, v in facts.items():
            lines.append(f'- {k}：{v}')
    faq = kb.get('faq', []) or []
    if faq:
        lines.append('\n【常见问答】')
        for item in faq:
            lines.append(f'- 问：{item.get("q", "")} 答：{item.get("a", "")}')
    lines.append('\n回答要求：只用中文、口语化、简短；资料中没有的内容如实说明不确定，不要编造。')
    return '\n'.join(lines)


def reload_kb(path: str = ''):
    """清除知识库进程缓存；path 为空则清空全部。

    在线编辑并保存 huadun_kb.json 后调用，无需重启 app.py 即可让下一次问答读到新内容。
    """
    if path:
        _kb_cache.pop(path, None)
    else:
        _kb_cache.clear()
    logger.info(f"[KB] cache cleared for: {path or 'ALL'}")

# Built-in LLM providers. Each exposes an OpenAI-compatible chat completions
# endpoint; the active one is selected with --llm_provider (default: dashscope).
LLM_PROVIDERS = {
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
    "orcarouter": {
        "api_key_env": "ORCAROUTER_API_KEY",
        "base_url": "https://api.orcarouter.ai/v1",
        "default_model": "orcarouter/auto",
    },
    "zhipu": {
        "api_key_env": "ZHIPUAI_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",   # 通用(按量付费)端点；Coding 套餐需走专属 /api/coding/paas/v4
        "default_model": "GLM-5.3-FlashX",  # 默认走免费档；精确 id 以开放平台模型详情页示例代码为准
        "fallback_model": "glm-5.3-flash",  # 免费额度耗尽/报错时自动回退的按量付费档
    },
    "minimax": {
        "api_key_env": "MINIMAX_API_KEY",
        "base_url": "https://api.minimaxi.com/v1",   # 国内站；国际站改 https://api.minimax.io/v1（Key 域名需与 base_url 一致）
        "default_model": "MiniMax-M2",                # 以 platform 控制台模型列表实际 id 为准
    },
}


def _llm_provider(opt) -> str:
    """Return the configured provider name, defaulting to dashscope."""
    return getattr(opt, 'llm_provider', 'dashscope') or 'dashscope'


def _llm_client(opt):
    """Create the OpenAI-compatible client for the configured provider."""
    from openai import OpenAI
    cfg = LLM_PROVIDERS.get(_llm_provider(opt), LLM_PROVIDERS['dashscope'])
    return OpenAI(
        api_key=os.getenv(cfg['api_key_env']),
        base_url=cfg['base_url'],
    )


def _llm_model(opt) -> str:
    """Resolve the model name, falling back to the provider default."""
    cfg = LLM_PROVIDERS.get(_llm_provider(opt), LLM_PROVIDERS['dashscope'])
    return getattr(opt, 'llm_model', '') or cfg['default_model']


def _llm_fallback_model(opt) -> str:
    """Return the provider's fallback model (empty if not configured)."""
    cfg = LLM_PROVIDERS.get(_llm_provider(opt), LLM_PROVIDERS['dashscope'])
    return cfg.get('fallback_model', '')


def _stream_llm(client, model, system_prompt, message, avatar_session, datainfo):
    """执行一次流式补全并逐句推送。

    返回 (ok, produced)：
      ok       - 请求是否正常完成（未在 create/迭代阶段抛错）；
      produced - 是否已推送过有效文本（用于判断能否安全重试，避免重复播报）。
    异常不向上抛，由调用方根据返回值决定是否回退。
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': system_prompt},
                      {'role': 'user', 'content': message}],
            stream=True,
            # Display token usage in the last line of the streamed response.
            stream_options={"include_usage": True}
        )
    except Exception as e:
        logger.warning(f"[LLM] create() failed on model={model}: {e}")
        return False, False

    produced = False
    result = ""
    first = True
    start = time.perf_counter()
    try:
        for chunk in completion:
            if len(chunk.choices) > 0:
                if first:
                    logger.info(f"llm Time to first chunk: {time.perf_counter()-start}s")
                    first = False
                msg = chunk.choices[0].delta.content
                if msg is None:
                    continue
                produced = True
                lastpos = 0
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：；":
                        result = result + msg[lastpos:i+1]
                        lastpos = i+1
                        if len(result) > 10:
                            logger.info(result)
                            avatar_session.put_msg_txt(result, datainfo)
                            result = ""
                result = result + msg[lastpos:]
    except Exception as e:
        logger.warning(f"[LLM] streaming failed on model={model} (produced={produced}): {e}")
        return False, produced
    if result:
        avatar_session.put_msg_txt(result, datainfo)
        produced = True
    logger.info(f"llm Time to last chunk: {time.perf_counter()-start}s")
    return True, produced


def llm_response(message,avatar_session:'BaseAvatar',datainfo:dict={}):
    try:
        opt = avatar_session.opt
        start = time.perf_counter()
        client = _llm_client(opt)
        model = _llm_model(opt)
        system_prompt = _build_system_prompt(opt)
        logger.info(f"llm Time init: {time.perf_counter()-start}s,{message}")

        ok, produced = _stream_llm(client, model, system_prompt, message, avatar_session, datainfo)

        # 回退：仅当主模型失败或空输出、且尚未推送任何文本时触发，避免重复播报
        fb = _llm_fallback_model(opt)
        if fb and fb != model and not produced:
            logger.warning(f"[LLM] model {model} ok={ok} produced=False -> fallback {fb}")
            _stream_llm(client, fb, system_prompt, message, avatar_session, datainfo)
        elif not ok and not produced:
            logger.error(f"[LLM] model {model} failed and no fallback available")
    except Exception as e:
        logger.exception('llm exception:')
        return
