###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
###############################################################################

import json
import asyncio
from aiohttp import web

from utils.logger import logger


# ─── 路由工具函数 ──────────────────────────────────────────────────────────

def json_ok(data=None):
    """返回成功 JSON 响应"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """返回错误 JSON 响应"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager
from server.avatar_routes import setup_avatar_routes

def get_session(request, sessionid: str):
    """从 app 中获取 session 实例"""
    return session_manager.get_session(sessionid)


# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """文本输入（echo/chat 模式），支持 voice/emotion 参数"""
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            datainfo['tts'] = params.get('tts')

        if params['type'] == 'echo':
            avatar_session.put_msg_txt(params['text'], datainfo)
        elif params['type'] == 'chat':
            llm_response = request.app.get("llm_response")
            if llm_response:
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params['text'], avatar_session, datainfo
                )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """打断当前说话"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as e:
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """上传音频文件"""
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.put_audio_file(filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def set_audiotype(request):
    """设置自定义状态（动作编排）"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def set_mouse(request):
    """ mouthdict: web 端上报鼠标坐标，驱动 GazeScheduler 视线跟随。非 mouthdict 无设卡→ no-op OK。"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if hasattr(avatar_session, "set_mouse_target"):
            avatar_session.set_mouse_target(params.get('x', 0), params.get('y', 0))
        return json_ok()
    except Exception as e:
        logger.exception('set_mouse exception:')
        return json_error(str(e))


async def set_mood(request):
    """mouthdict: 情绪切换（P1 可配置/环境变量硬切，P2 接 LLM 意图总线）。"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if hasattr(avatar_session, "set_mood"):
            avatar_session.set_mood(
                params.get('mood', 'neutral'),
                fade_ms=params.get('fade_ms'),
            )
        return json_ok()
    except Exception as e:
        logger.exception('set_mood exception:')
        return json_error(str(e))


async def set_state(request):
    """mouthdict: 交互状态切换 (speaking/listening/idle)。客户说话时前端自动切 listening。"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        state = params.get('state', 'idle')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if hasattr(avatar_session, "set_interaction_state"):
            avatar_session.set_interaction_state(state)
        return json_ok()
    except Exception as e:
        logger.exception('set_state exception:')
        return json_error(str(e))


async def record(request):
    """录制控制"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params['type'] == 'start_record':
            avatar_session.start_recording()
        elif params['type'] == 'end_record':
            avatar_session.stop_recording()
        return json_ok()
    except Exception as e:
        logger.exception('record exception:')
        return json_error(str(e))


async def welcome(request):
    """迎宾序列：服务端编排完整迎宾流程（敬礼→招手→微笑→问候→聆听）。

    Body:
        sessionid: str  — 会话 ID
        text: str       — 问候词（可选，空则用 KB greeting）
        salute_ms: int  — 敬礼动画时长 ms（默认 2500）
        beckon_ms: int  — 招手动画时长 ms（默认 1500）
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        greet_text = params.get('text', '').strip()
        if not greet_text:
            # 回退读 KB greeting
            import os, json as _json
            opt = request.app.get("opt")
            kb_path = getattr(opt, "kb_path", "") if opt else ""
            if not kb_path:
                kb_path = os.path.join("data", "huadun_kb.json")
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    greet_text = _json.load(f).get("greeting", "欢迎您！")
            except Exception:
                greet_text = "欢迎您！有什么可以帮您？"

        salute_ms = int(params.get('salute_ms', 2500))
        beckon_ms = int(params.get('beckon_ms', 1500))

        async def _run_sequence():
            """后台异步执行迎宾序列。"""
            try:
                # 1) 敬礼
                avatar_session.set_custom_state(2)
                await asyncio.sleep(salute_ms / 1000.0)

                # 2) 招手
                avatar_session.set_custom_state(3)
                await asyncio.sleep(beckon_ms / 1000.0)

                # 3) 微笑表情
                if hasattr(avatar_session, 'set_mood'):
                    avatar_session.set_mood('happy', fade_ms=300)

                # 4) 问候语音（echo 模式 TTS）
                avatar_session.put_msg_txt(greet_text, {})

                # 5) 等问候说完后进入聆听态
                # （简单延时估计，实际可监听 SSE end 事件）
                est_speech_s = max(3.0, len(greet_text) * 0.35)
                await asyncio.sleep(est_speech_s)
                if hasattr(avatar_session, 'set_interaction_state'):
                    avatar_session.set_interaction_state('listening')

                logger.info(f"welcome sequence completed for session={sessionid}")
            except Exception as e:
                logger.exception(f"welcome sequence error: {e}")

        # 启动后台序列，立即返回
        asyncio.ensure_future(_run_sequence())
        return json_ok(data={"greeting": greet_text, "salute_ms": salute_ms, "beckon_ms": beckon_ms})
    except Exception as e:
        logger.exception('welcome exception:')
        return json_error(str(e))


async def is_speaking(request):
    """查询是否正在说话"""
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())

async def sse_handler(request):
    """SSE 事件流，推送服务器状态更新到客户端"""
    sessionid = request.query.get('sessionid', '')
    avatar_session = session_manager.get_session(sessionid)
    if avatar_session is None:
        return json_error("session not found")

    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        }
    )
    await response.prepare(request)

    import queue
    msgqueue = queue.Queue()
    avatar_session.add_msgqueue(msgqueue)

    try:
        while True:
            try:
                msg = msgqueue.get_nowait()
                await response.write(f"data: {msg}\n\n".encode('utf-8'))
            except queue.Empty:
                await asyncio.sleep(0.01)
    except (asyncio.CancelledError, ConnectionResetError):
        logger.info('SSE connection closed for session: %s', sessionid)
    finally:
        if msgqueue in avatar_session.msgqueues:
            avatar_session.msgqueues.remove(msgqueue)

    return response


async def get_kb(request):
    """返回公司知识库 JSON（供迎宾前端读取问候语/介绍/人设）。

    优先读 opt.kb_path；未配置则回退 data/huadun_kb.json。"""
    try:
        import os
        opt = request.app.get("opt")
        path = getattr(opt, "kb_path", "") if opt else ""
        if not path:
            path = os.path.join("data", "huadun_kb.json")
        if not os.path.exists(path):
            return json_ok(data={})
        with open(path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "ok", "data": kb}, ensure_ascii=False),
        )
    except Exception as e:
        logger.exception('get_kb exception:')
        return json_error(str(e))


async def save_kb(request):
    """保存/编辑知识库 JSON 到 kb_path，并热加载（清 llm 进程缓存），无需重启。

    body 为完整知识库对象（与 huadun_kb.json 同构）。写入前先备份为 *.bak。"""
    try:
        import os
        import shutil
        kb = await request.json()
        if not isinstance(kb, dict):
            return json_error("知识库必须是一个 JSON 对象")
        # 基本结构校验，避免把文件写坏
        for key in ("persona", "greeting", "facts", "faq"):
            if key not in kb:
                return json_error(f"知识库缺少必要字段: {key}")

        opt = request.app.get("opt")
        path = getattr(opt, "kb_path", "") if opt else ""
        if not path:
            path = os.path.join("data", "huadun_kb.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            try:
                shutil.copy2(path, path + ".bak")
            except Exception:
                logger.warning('save_kb: 备份旧知识库失败，继续写入')
        with open(path, "w", encoding="utf-8") as f:
            json.dump(kb, f, ensure_ascii=False, indent=2)

        # 热加载：下次问答重新从文件读取
        try:
            import llm
            llm.reload_kb(path)
        except Exception as e:
            logger.warning(f'save_kb: 热加载缓存清理失败 {e}')
        logger.info('save_kb: 知识库已保存并热加载: %s', path)
        return json_ok(data={"path": path})
    except Exception as e:
        logger.exception('save_kb exception:')
        return json_error(str(e))


async def admin_config(request):
    """Admin: 获取全局配置参数"""
    try:
        opt = request.app.get("opt")
        if opt:
            return json_ok(data={"config": vars(opt)})
        return json_error("Config not found")
    except Exception as e:
        logger.exception('admin_config exception:')
        return json_error(str(e))


async def admin_sessions(request):
    """Admin: 获取活跃的会话及其配置"""
    try:
        sessions_info = []
        for sid, avatar_session in session_manager.sessions.items():
            if avatar_session:
                s_opt = getattr(avatar_session, 'opt', None)
                s_data = {
                    "sessionid": sid,
                    "speaking": avatar_session.is_speaking() if hasattr(avatar_session, 'is_speaking') else False,
                    "recording": getattr(avatar_session, 'recording', False),
                }
                if s_opt:
                    s_data.update({
                        "model": getattr(s_opt, "model", ""),
                        "avatar_id": getattr(s_opt, "avatar_id", ""),
                        "REF_FILE": getattr(s_opt, "REF_FILE", ""),
                        "transport": getattr(s_opt, "transport", ""),
                        "batch_size": getattr(s_opt, "batch_size", 0),
                        "customopt": getattr(s_opt, "customopt", []),
                    })
                sessions_info.append(s_data)
        return json_ok(data={"sessions": sessions_info})
    except Exception as e:
        logger.exception('admin_sessions exception:')
        return json_error(str(e))


# ─── 路由注册 ──────────────────────────────────────────────────────────────

async def index(request):
    """默认首页重定向"""
    opt = request.app.get("opt")
    pagename = 'index.html'
    if opt and opt.transport == 'rtmp':
        pagename = 'rtmpapi.html'
    elif opt and opt.transport == 'rtcpush':
        pagename = 'rtcpushapi.html'
    raise web.HTTPFound(f'/{pagename}')


def setup_routes(app):
    """注册所有路由到 aiohttp app"""
    app.router.add_get("/", index)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/mouse", set_mouse)
    app.router.add_post("/mood", set_mood)
    app.router.add_post("/set_state", set_state)
    app.router.add_post("/welcome", welcome)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_get("/api/kb", get_kb)
    app.router.add_post("/api/kb", save_kb)
    app.router.add_get("/api/admin/config", admin_config)
    app.router.add_get("/api/admin/sessions", admin_sessions)
    app.router.add_get('/sse', sse_handler)

    # ── Local ASR endpoint (SenseVoice/FunASR) ── Issue #604 ──
    try:
        from server.asr_server import asr_websocket_handler, is_funasr_available
        if is_funasr_available():
            app.router.add_get("/api/asr", asr_websocket_handler)
            logger.info("[ASR] Local SenseVoice ASR endpoint enabled at /api/asr")
        else:
            logger.info("[ASR] funasr not installed — local ASR endpoint disabled "
                        "(pip install funasr modelscope)")
    except Exception as e:
        logger.warning(f"[ASR] Failed to register ASR endpoint: {e}")

    # 注册 avatar 生成相关的路由
    setup_avatar_routes(app)

    app.router.add_static('/', path='web')
