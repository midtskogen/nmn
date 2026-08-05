(function () {
    'use strict';

    function initTheme() {
        const body = document.body;
        const canvas = document.getElementById('starfield');
        const themeRadios = document.querySelectorAll('input[name="nmn-theme"]');
        const themes = {
            classic: { stars: false, starColor: '#ffffff', shootingStarColor: '#ffffff', bg: null },
            night:   { stars: true,  starColor: '#ffffff', shootingStarColor: '#ffd166', radiant: { x: 0.2, y: 0.2 }, bg: { type: 'radial', stops: [[0, '#1b2735'], [1, '#090a0f']] } }
        };

        let starfield = null;
        if (canvas) starfield = initStarfield(canvas);

        function applyTheme(name) {
            const theme = themes[name] || themes.classic;
            body.classList.forEach(cls => { if (cls.startsWith('theme-')) body.classList.remove(cls); });
            body.classList.remove('theme-dark');
            body.classList.add(`theme-${name}`);
            if (name !== 'classic') body.classList.add('theme-dark');
            localStorage.setItem('nmn-meteor-theme', name);
            themeRadios.forEach(radio => {
                const checked = radio.value === name;
                radio.checked = checked;
                const label = radio.closest('label');
                if (label) label.classList.toggle('active', checked);
            });
            if (starfield) starfield.setOptions(theme);
        }

        themeRadios.forEach(radio => {
            radio.addEventListener('change', () => { if (radio.checked) applyTheme(radio.value); });
        });

        const saved = localStorage.getItem('nmn-meteor-theme') || 'classic';
        applyTheme(saved);
    }

    function initStarfield(canvas) {
        const ctx = canvas.getContext('2d');
        let width = 0, height = 0;
        let stars = [];
        let shootingStars = [];
        let opts = { stars: false, starColor: '#ffffff', shootingStarColor: '#ffffff', bg: null };
        let rafId = null;

        const hexToRgb = (hex) => {
            const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            return m ? `${parseInt(m[1], 16)}, ${parseInt(m[2], 16)}, ${parseInt(m[3], 16)}` : '255, 255, 255';
        };

        function resize() {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        }

        function createStars() {
            stars = [];
            const count = Math.floor((width * height) / 3500);
            for (let i = 0; i < count; i++) {
                // Bias toward faint stars: most are small and dim.
                const brightness = Math.pow(Math.random(), 2);
                stars.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    r: 0.2 + brightness * 1.4,
                    baseAlpha: 0.2 + brightness * 0.6,
                    phase: Math.random() * Math.PI * 2,
                    speed: Math.random() * 0.02 + 0.005,
                    state: 'stable',
                    stateTimer: Math.floor(Math.random() * 240) + 60
                });
            }
        }

        function spawnShootingStar() {
            if (!opts.stars) return;
            let x, y, angle;
            if (opts.radiant) {
                const rx = width * opts.radiant.x;
                const ry = height * opts.radiant.y;
                // Start somewhere along a random ray from the radiant point.
                angle = Math.random() * Math.PI * 2;
                const distance = Math.random() * Math.max(width, height) * 0.8;
                x = rx + Math.cos(angle) * distance;
                y = ry + Math.sin(angle) * distance;
            } else {
                x = Math.random() * width;
                y = Math.random() * height * 0.6;
                angle = Math.PI / 4 + Math.random() * Math.PI / 6;
            }
            shootingStars.push({
                x: x,
                y: y,
                len: Math.random() * 80 + 40,
                speed: Math.random() * 12 + 8,
                angle: angle,
                life: 1
            });
        }

        function draw() {
            if (opts.bg) {
                let grd;
                if (opts.bg.type === 'radial') {
                    grd = ctx.createRadialGradient(width / 2, height, 0, width / 2, height / 2, Math.max(width, height));
                } else {
                    grd = ctx.createLinearGradient(0, 0, 0, height);
                }
                opts.bg.stops.forEach(([pos, color]) => grd.addColorStop(pos, color));
                ctx.fillStyle = grd;
                ctx.fillRect(0, 0, width, height);
            } else {
                ctx.clearRect(0, 0, width, height);
            }
            if (opts.stars) {
                const [r, g, b] = hexToRgb(opts.starColor).split(',').map(s => parseInt(s.trim(), 10));
                stars.forEach(star => {
                    star.stateTimer--;
                    if (star.stateTimer <= 0) {
                        if (star.state === 'stable') {
                            star.state = 'flicker';
                            star.stateTimer = Math.floor(Math.random() * 30) + 15;
                        } else {
                            star.state = 'stable';
                            star.stateTimer = Math.floor(Math.random() * 240) + 60;
                        }
                    }
                    star.phase += star.speed * (star.state === 'flicker' ? 6 : 1);
                    let flicker = 0;
                    if (star.state === 'flicker') {
                        flicker = (Math.random() - 0.5) * 0.35;
                    }
                    const slowPulse = Math.sin(star.phase) * 0.04;
                    let alpha = star.baseAlpha + slowPulse + flicker;
                    if (alpha < 0) alpha = 0;
                    if (alpha > 1) alpha = 1;
                    ctx.beginPath();
                    ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
                    ctx.fill();
                });

                const [sr, sg, sb] = hexToRgb(opts.shootingStarColor).split(',').map(s => parseInt(s.trim(), 10));
                for (let i = shootingStars.length - 1; i >= 0; i--) {
                    const s = shootingStars[i];
                    s.x += Math.cos(s.angle) * s.speed;
                    s.y += Math.sin(s.angle) * s.speed;
                    s.life -= 0.02;
                    const tailX = s.x - Math.cos(s.angle) * s.len;
                    const tailY = s.y - Math.sin(s.angle) * s.len;
                    const grad = ctx.createLinearGradient(s.x, s.y, tailX, tailY);
                    grad.addColorStop(0, `rgba(${sr}, ${sg}, ${sb}, ${Math.max(0, s.life)})`);
                    grad.addColorStop(1, `rgba(${sr}, ${sg}, ${sb}, 0)`);
                    ctx.strokeStyle = grad;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(s.x, s.y);
                    ctx.lineTo(tailX, tailY);
                    ctx.stroke();
                    if (s.life <= 0 || s.x > width + s.len || s.y > height + s.len) {
                        shootingStars.splice(i, 1);
                    }
                }
                if (Math.random() < 0.02) spawnShootingStar();
            }
            rafId = requestAnimationFrame(draw);
        }

        function setOptions(options) {
            opts = options || opts;
            if (opts.stars) {
                resize();
                createStars();
            } else {
                shootingStars = [];
            }
        }

        window.addEventListener('resize', () => { resize(); createStars(); });
        resize();
        createStars();
        draw();

        return { setOptions };
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTheme);
    } else {
        initTheme();
    }
})();
