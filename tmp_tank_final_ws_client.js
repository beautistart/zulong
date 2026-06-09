const fs = require("node:fs")
const path = require("node:path")

const logPath = path.resolve("tmp_zulong_tank_final_events.jsonl")
const summaryPath = path.resolve("tmp_zulong_tank_final_summary.json")
try {
	fs.unlinkSync(logPath)
} catch {}
try {
	fs.unlinkSync(summaryPath)
} catch {}

const url = "ws://127.0.0.1:8090/ws"
const ws = new WebSocket(url)
const start = Date.now()
const counters = {}
let final = null
const approvals = []
const statuses = []
const taskGraphs = []
const fileEvents = []
const ideEvents = []
const approvedIds = new Set()

function log(obj) {
	fs.appendFileSync(logPath, JSON.stringify({ dt: Date.now() - start, ...obj }) + "\n", "utf8")
}

function send(type, payload) {
	if (type === "CHAT_MESSAGE") {
		ws.send(JSON.stringify({
			type,
			...payload,
			session_id: payload.session_id || `tank_final_session_${start}`,
			request_id: payload.request_id || `tank_final_${start}`,
			ts: Date.now() / 1000,
		}))
		return
	}
	ws.send(JSON.stringify({
		type,
		payload,
		session_id: `tank_final_session_${start}`,
		request_id: `tank_final_${start}`,
		ts: Date.now() / 1000,
	}))
}

function taskText() {
	return `请执行一个真实复杂开发任务，用来测试祖龙系统复杂任务执行链路。

任务目标：让祖龙在 D:\\AI\\project 这个父目录下创建英文文件夹 “tank”，并且在 D:\\AI\\project\\tank 文件夹写一个可直接在 Web 端运行的像素风“坦克大战”小游戏。

执行要求：
1. 最终项目目录必须是 D:\\AI\\project\\tank。不要创建中文目录，不要创建 D:\\AI\\project\\.zulong 或 .zlong 目录。
2. 游戏至少包含 index.html、style.css、game.js、README.md。
3. 游戏为像素风 Web 端运行，玩家坦克可移动和射击，敌方坦克会移动/射击或追踪，包含墙体/障碍、生命、分数、关卡或波次、开始/暂停/重新开始。
4. 必须使用 VS Code/IDE 桥进行文件创建和编辑；如果出现审批弹窗，必须暂停等待用户审批，审批通过后再继续，不要继续假装执行。
5. 执行过程中持续汇报任务状态、当前步骤和下一步，让用户能判断任务正在执行、完成、受阻或疑似卡死。
6. 完成后说明实际创建文件路径、如何在浏览器运行、VS Code 调用/审批暂停/任务恢复状态是否正常。

这是复杂任务测试，请严格对齐 TSD 的任务生命周期汇报、审批暂停、路径隔离、工具调用和任务恢复设计。`
}

ws.addEventListener("open", () => {
	log({ kind: "open" })
	send("CHAT_MESSAGE", {
		text: taskText(),
		conversation_id: `tank_final_conv_${start}`,
		request_id: `tank_final_${start}`,
	})
})

ws.addEventListener("message", (event) => {
	let data
	try {
		data = JSON.parse(event.data.toString())
	} catch {
		data = { raw: event.data.toString() }
	}
	const type = data.type || data.action || "unknown"
	counters[type] = (counters[type] || 0) + 1
	const payload = data.payload || data
	log({ kind: "event", type, payload })

	if (type === "TASK_EXECUTION_STATUS") statuses.push(payload)
	if (type === "TASK_GRAPH_UPDATE") taskGraphs.push(payload)
	if (type === "IDE_APPROVAL_STATUS" || type === "approval_required" || type === "ide:approval_status") approvals.push(payload)
	if (type === "IDE_FILE_CHANGED" || type === "ide:file_changed") fileEvents.push(payload)
	if (type.startsWith("IDE") || type.startsWith("ide:") || type.startsWith("ide_")) ideEvents.push({ type, payload })

	if (type === "TASK_COMPLETE" || type === "task_complete" || type === "TASK_ERROR" || type === "task_error") {
		final = { type, payload, dt: Date.now() - start }
		setTimeout(() => ws.close(), 1000)
	}

	for (const item of approvals) {
		const req = item.payload || item
		if (req.approval_id && !approvedIds.has(req.approval_id)) {
			approvedIds.add(req.approval_id)
			log({ kind: "auto_approve", approval_id: req.approval_id, tool: req.tool_name, summary: req.action_summary })
			send("ide_approval_result", {
				approval_id: req.approval_id,
				approved: true,
				ide_session_id: req.ide_session_id,
				workspace_path: req.workspace_path,
				tool_name: req.tool_name,
				action_summary: req.action_summary,
			})
		}
	}
})

ws.addEventListener("close", () => finish("close"))
ws.addEventListener("error", (event) => {
	log({ kind: "error", message: event.message || "websocket error" })
	finish("error")
})

setTimeout(() => {
	try {
		ws.close()
	} catch {}
	finish("timeout")
}, 900000)

let finished = false
function finish(reason) {
	if (finished) return
	finished = true
	const tankPath = "D:/AI/project/tank"
	const exists = fs.existsSync(tankPath)
	const files = exists ? fs.readdirSync(tankPath, { withFileTypes: true }).map((d) => d.name) : []
	const dotZulong = fs.existsSync("D:/AI/project/.zulong")
	const summary = {
		reason,
		elapsed_ms: Date.now() - start,
		counters,
		final,
		approvals: approvals.map((a) => ({
			tool: a.tool_name || a.payload?.tool_name,
			summary: a.action_summary || a.payload?.action_summary,
			id: a.approval_id || a.payload?.approval_id,
		})),
		lastStatuses: statuses.slice(-8),
		taskGraphCount: taskGraphs.length,
		fileEvents,
		ideEventTypes: [...new Set(ideEvents.map((e) => e.type))],
		tankExists: exists,
		files,
		dotZulongExists: dotZulong,
		logPath,
	}
	fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2), "utf8")
	console.log(JSON.stringify(summary, null, 2))
}
