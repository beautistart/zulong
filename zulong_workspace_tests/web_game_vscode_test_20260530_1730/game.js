/* ===== 星尘接球 — 游戏核心逻辑 ===== */

(function () {
    "use strict";

    // ─── DOM 引用 ────────────────────────────
    const canvas = document.getElementById("game-canvas");
    const ctx = canvas.getContext("2d");

    const scoreEl = document.getElementById("score");
    const livesEl = document.getElementById("lives");
    const levelEl = document.getElementById("level");
    const messageEl = document.getElementById("message");

    const btnStart = document.getElementById("btn-start");
    const btnPause = document.getElementById("btn-pause");
    const btnRestart = document.getElementById("btn-restart");

    // ─── 游戏常量 ────────────────────────────
    const CANVAS_W = canvas.width;   // 600
    const CANVAS_H = canvas.height;  // 500
    const PADDLE_W = 100;
    const PADDLE_H = 14;
    const PADDLE_Y = CANVAS_H - 50;
    const STAR_RADIUS = 10;
    const BASE_STAR_SPEED = 2.0;
    const SPEED_INCREMENT = 0.4;
    const LEVEL_UP_SCORE = 200;

    // ─── 游戏状态 ────────────────────────────
    let paddleX = (CANVAS_W - PADDLE_W) / 2;
    let stars = [];
    let score = 0;
    let lives = 3;
    let level = 1;
    let gameRunning = false;
    let gamePaused = false;
    let animFrameId = null;
    let spawnTimer = 0;
    let spawnInterval = 45;
    let frameCount = 0;

    // ─── 辅助函数 ────────────────────────────
    function rand(min, max) {
        return Math.random() * (max - min) + min;
    }

    function updateHUD() {
        scoreEl.textContent = score;
        livesEl.textContent = lives;
        levelEl.textContent = level;
    }

    function setMessage(text, isGood = true) {
        messageEl.textContent = text;
        messageEl.style.color = isGood ? "#ffcc80" : "#ff6e6e";
    }

    // ─── 星星类 ──────────────────────────────
    function createStar() {
        const x = rand(STAR_RADIUS, CANVAS_W - STAR_RADIUS);
        const speed = BASE_STAR_SPEED + (level - 1) * SPEED_INCREMENT;
        const hue = rand(35, 55);
        const color = `hsl(${hue}, 90%, ${rand(55, 75)}%)`;
        return {
            x: x,
            y: -STAR_RADIUS,
            radius: rand(6, STAR_RADIUS),
            speed: speed * rand(0.8, 1.3),
            color: color,
            twinkle: rand(0, Math.PI * 2)
        };
    }

    // ─── 绘制函数 ────────────────────────────
    function drawBackground() {
        ctx.fillStyle = "#080820";
        ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

        ctx.fillStyle = "rgba(255,255,255,0.4)";
        for (let i = 0; i < 60; i++) {
            const sx = (i * 137.5 + 50) % CANVAS_W;
            const sy = (i * 97.3 + 30) % CANVAS_H;
            const sr = (i % 3) + 0.8;
            ctx.beginPath();
            ctx.arc(sx, sy, sr, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function drawPaddle() {
        const gradient = ctx.createLinearGradient(paddleX, PADDLE_Y, paddleX, PADDLE_Y + PADDLE_H);
        gradient.addColorStop(0, "#7ec8f8");
        gradient.addColorStop(0.5, "#4aa8e8");
        gradient.addColorStop(1, "#1a68b8");

        ctx.shadowColor = "rgba(100,180,255,0.8)";
        ctx.shadowBlur = 18;
        ctx.fillStyle = gradient;
        ctx.beginPath();
        const r = PADDLE_H / 2;
        ctx.moveTo(paddleX + r, PADDLE_Y);
        ctx.lineTo(paddleX + PADDLE_W - r, PADDLE_Y);
        ctx.arc(paddleX + PADDLE_W - r, PADDLE_Y + r, r, -Math.PI / 2, Math.PI / 2);
        ctx.lineTo(paddleX + r, PADDLE_Y + PADDLE_H);
        ctx.arc(paddleX + r, PADDLE_Y + r, r, Math.PI / 2, -Math.PI / 2);
        ctx.closePath();
        ctx.fill();
        ctx.shadowBlur = 0;
    }

    function drawStars() {
        for (const star of stars) {
            star.twinkle += 0.08;
            const alpha = 0.6 + 0.4 * Math.sin(star.twinkle);

            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.shadowColor = star.color;
            ctx.shadowBlur = 12;

            ctx.fillStyle = star.color;
            ctx.beginPath();
            const cx = star.x;
            const cy = star.y;
            const outerR = star.radius;
            const innerR = star.radius * 0.4;
            const spikes = 5;
            for (let i = 0; i < spikes * 2; i++) {
                const r = i % 2 === 0 ? outerR : innerR;
                const angle = (i * Math.PI) / spikes - Math.PI / 2;
                const px = cx + Math.cos(angle) * r;
                const py = cy + Math.sin(angle) * r;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.closePath();
            ctx.fill();

            ctx.restore();
        }
    }

    function drawAll() {
        drawBackground();
        drawStars();
        drawPaddle();
    }

    // ─── 碰撞检测 ────────────────────────────
    function checkCollision(star) {
        const starBottom = star.y + star.radius;
        const starLeft = star.x - star.radius;
        const starRight = star.x + star.radius;

        if (
            starBottom >= PADDLE_Y &&
            starBottom <= PADDLE_Y + PADDLE_H + 6 &&
            starRight > paddleX &&
            starLeft < paddleX + PADDLE_W
        ) {
            return true;
        }
        return false;
    }

    // ─── 游戏逻辑更新 ────────────────────────
    function update() {
        if (!gameRunning || gamePaused) return;

        frameCount++;

        spawnTimer++;
        if (spawnTimer >= spawnInterval) {
            spawnTimer = 0;
            stars.push(createStar());
            spawnInterval = Math.max(18, 45 - (level - 1) * 3);
        }

        for (let i = stars.length - 1; i >= 0; i--) {
            const star = stars[i];
            star.y += star.speed;

            if (checkCollision(star)) {
                stars.splice(i, 1);
                score += 10;
                updateHUD();

                const newLevel = Math.floor(score / LEVEL_UP_SCORE) + 1;
                if (newLevel > level) {
                    level = newLevel;
                    updateHUD();
                    setMessage(`🎉 升级！当前等级 ${level}`, true);
                }
                continue;
            }

            if (star.y - star.radius > CANVAS_H) {
                stars.splice(i, 1);
                lives--;
                updateHUD();

                if (lives <= 0) {
                    gameOver();
                    return;
                }
                setMessage(`💔 失去一颗星！剩余生命: ${lives}`, false);
            }
        }
    }

    function gameLoop() {
        update();
        drawAll();
        animFrameId = requestAnimationFrame(gameLoop);
    }

    // ─── 游戏控制 ────────────────────────────
    function startGame() {
        if (gameRunning && !gamePaused) return;
        if (lives <= 0) {
            resetGame();
        }

        gameRunning = true;
        gamePaused = false;
        btnStart.textContent = "⏸ 暂停";
        btnPause.disabled = false;
        setMessage("🌟 游戏进行中...移动鼠标接住星星！", true);

        if (!animFrameId) {
            gameLoop();
        }
    }

    function pauseGame() {
        if (!gameRunning) return;
        gamePaused = !gamePaused;
        if (gamePaused) {
            btnStart.textContent = "▶ 继续";
            btnPause.textContent = "▶ 继续";
            setMessage("⏸ 已暂停", true);
        } else {
            btnStart.textContent = "⏸ 暂停";
            btnPause.textContent = "⏸ 暂停";
            setMessage("🌟 游戏继续！", true);
        }
    }

    function resetGame() {
        stars = [];
        score = 0;
        lives = 3;
        level = 1;
        gameRunning = false;
        gamePaused = false;
        spawnTimer = 0;
        spawnInterval = 45;
        frameCount = 0;
        paddleX = (CANVAS_W - PADDLE_W) / 2;
        btnStart.textContent = "▶ 开始游戏";
        btnPause.disabled = true;
        btnPause.textContent = "⏸ 暂停";
        updateHUD();
        setMessage("🔄 已重置，点击开始！", true);
        drawAll();
    }

    function gameOver() {
        gameRunning = false;
        gamePaused = false;
        btnStart.textContent = "▶ 重新开始";
        btnPause.disabled = true;
        setMessage(`💀 游戏结束！最终得分: ${score} | 等级: ${level}`, false);

        if (animFrameId) {
            cancelAnimationFrame(animFrameId);
            animFrameId = null;
        }
    }

    // ─── 鼠标/触摸控制 ────────────────────────
    canvas.addEventListener("mousemove", function (e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = CANVAS_W / rect.width;
        const mouseX = (e.clientX - rect.left) * scaleX;
        paddleX = Math.max(0, Math.min(CANVAS_W - PADDLE_W, mouseX - PADDLE_W / 2));
    });

    canvas.addEventListener("touchmove", function (e) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const scaleX = CANVAS_W / rect.width;
        const touchX = (e.touches[0].clientX - rect.left) * scaleX;
        paddleX = Math.max(0, Math.min(CANVAS_W - PADDLE_W, touchX - PADDLE_W / 2));
    }, { passive: false });

    // ─── 按钮事件 ────────────────────────────
    btnStart.addEventListener("click", function () {
        if (!gameRunning || gamePaused) {
            startGame();
        } else {
            pauseGame();
        }
    });

    btnPause.addEventListener("click", pauseGame);
    btnRestart.addEventListener("click", resetGame);

    // ─── 键盘快捷键 ──────────────────────────
    document.addEventListener("keydown", function (e) {
        switch (e.key.toLowerCase()) {
            case " ":
            case "enter":
                e.preventDefault();
                if (!gameRunning || gamePaused) startGame();
                else pauseGame();
                break;
            case "r":
                if (!e.ctrlKey && !e.metaKey) {
                    resetGame();
                }
                break;
            case "arrowleft":
                paddleX = Math.max(0, paddleX - 25);
                break;
            case "arrowright":
                paddleX = Math.min(CANVAS_W - PADDLE_W, paddleX + 25);
                break;
        }
    });

    // ─── 初始化 ──────────────────────────────
    function init() {
        updateHUD();
        btnPause.disabled = true;
        setMessage("👆 移动鼠标控制托盘，点击「开始游戏」或按空格键！", true);
        drawAll();
    }

    init();
})();
